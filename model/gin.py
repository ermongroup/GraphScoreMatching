import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gnn import GraphNeuralNetwork
from model.mlp import MLP
from utils.graph_utils import add_self_loop_if_not_exists


def doubly_stochastic_norm(adjs_f, do_row_norm=True):
    """

    :param do_row_norm:
    :param adjs_f: B x N x N
    :return: normalized adjs_f: B x N x N
    """
    assert isinstance(adjs_f, torch.Tensor)
    # if len(adjs_f.size()) == 3:
    #     e_hat= adjs_f.unsqueeze(-1)
    # else:
    #     e_hat = adjs_f
    if do_row_norm:
        e_hat = adjs_f
        e_hat_row_sum = e_hat.sum(dim=2, keepdim=True)  # B x N x 1
        e_tilde = e_hat / e_hat_row_sum
    else:
        e_tilde = adjs_f
    e_tilde_col_sum = e_tilde.sum(dim=1, keepdim=True)  # B x 1 x N
    e = torch.bmm(e_tilde / e_tilde_col_sum, e_tilde.transpose(1, 2))
    return e


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features + 1, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # input: BS x N x F_in
        # W: F_in x F_out
        h = torch.matmul(input, self.W)
        # h: BS x N x F_out
        # batch_size = h.size(0)
        node_num = h.size(1)
        h_b = h.unsqueeze(-2).expand(-1, -1, node_num, -1)  # BS x N x N x F_out
        h_b_t = h_b.transpose(1, 2)  # BS x N x N x F_out

        a_input = torch.cat([h_b, h_b_t, adj.unsqueeze(-1)], dim=-1)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))

        # e: BS x N x N

        # attention = e.exp() * adj
        # new_adjs = doubly_stochastic_norm(attention, do_row_norm=True)


        # zero_vec = -9e15*torch.ones_like(e)
        # attention = torch.where(adj > 0, e, zero_vec)
        # check_adjs_symmetry(attention)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        attention = F.softmax(e, dim=-1)  #
        new_adjs = attention * adj
        h_prime = torch.matmul(new_adjs, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GIN(GraphNeuralNetwork):

    def __init__(self, feature_nums, dropout_p=0.5, out_dim=1, use_norm_layers=True,
                 attention_heads=None, channel_num=1, **kwargs):
        self._out_dim = out_dim
        self.channel_num = channel_num
        self.feature_nums = feature_nums
        hidden_num = 2 * max(feature_nums)

        def linear_with_leaky_relu(ii):
            return nn.Sequential(nn.Linear(feature_nums[ii], hidden_num),
                                 nn.LeakyReLU(),
                                 nn.Linear(hidden_num, out_dim))

        layer_n = len(feature_nums) - 1
        super().__init__(layer_n)
        self.use_norm_layers = use_norm_layers
        self.eps = nn.Parameter(torch.zeros(layer_n))
        if self.use_norm_layers:
            self.norm_layers = torch.nn.ModuleList()
        self.linear_prediction = torch.nn.ModuleList()
        self.layers = torch.nn.ModuleList()

        if attention_heads is not None and len(attention_heads) > 0:
            self.attentions = torch.nn.ModuleList()
            assert len(attention_heads) == layer_n
            attention_heads = [1] + attention_heads
        else:
            self.attentions = None

        for i in range(layer_n):
            if isinstance(attention_heads, list):
                assert feature_nums[i + 1] % attention_heads[i + 1] == 0
                attentions = [GraphAttentionLayer(feature_nums[i],
                                                  feature_nums[i + 1] // attention_heads[i + 1],
                                                  dropout=dropout_p,
                                                  alpha=0.02) for _ in range(attention_heads[i + 1])]
                mlp = MLP(num_layers=2, input_dim=feature_nums[i + 1] // attention_heads[i + 1],
                          hidden_dim=hidden_num,
                          output_dim=feature_nums[i + 1] // attention_heads[i + 1] // channel_num)
                attentions = torch.nn.ModuleList(attentions)
                self.attentions.append(attentions)
            else:
                mlp = MLP(num_layers=2, input_dim=feature_nums[i] * channel_num,
                          hidden_dim=hidden_num,
                          output_dim=feature_nums[i + 1])
            self.layers.append(mlp)
            if self.use_norm_layers:
                self.norm_layers.append(nn.BatchNorm1d(feature_nums[i + 1]))
                # self.norm_layers.append(
                #     ConditionalNorm1dPlus(num_features=feature_nums[i],
                #                                   num_classes=1))
            self.linear_prediction.append(linear_with_leaky_relu(i))
        self.linear_prediction.append(linear_with_leaky_relu(-1))

        self.dropout_p = dropout_p
        self.hidden = []

    def get_out_dim(self):
        return self._out_dim

    def _aggregate(self, x, adjs, node_flags, layer_k):
        batch_size = x.size(0)
        feature_num = x.size(-1)
        if self.use_norm_layers:
            x = self.norm_layers[layer_k](x.view(-1, feature_num)).contiguous().view(batch_size, -1, feature_num)
        if self.attentions is None:
            if len(adjs.shape) == 4:
                h = torch.matmul(adjs, x.unsqueeze(1))  # B x C x N x F
                h = h.permute(0, 2, 1, 3).contiguous().view(adjs.size(0), adjs.size(-1), -1)  # B x N x CF
            else:
                h = torch.bmm(adjs, x)
            h = h + self.eps[layer_k] * torch.cat([x]*self.channel_num, dim=-1)

            feature_num = h.size(-1)
            h = h.view(-1, feature_num)
            # print(h.size())
            # print(self.layers[layer_k])
            h = self.layers[layer_k](h)
            h = torch.tanh(h)

            h = h.view(batch_size, -1, h.size(-1))
        else:
            att_results = []
            for att in self.attentions[layer_k]:
                h = att(x, adjs)
                feature_num = h.size(-1)
                h = h.view(-1, feature_num)

                h = self.layers[layer_k](h)
                h = torch.relu(h)

                h = h.view(batch_size, -1, h.size(-1))
                att_results.append(h)
            h = torch.cat(att_results, dim=-1)

        self.hidden.append((h * node_flags.unsqueeze(-1)))

        return h

    def _combine(self, x, a, layer_k):
        return a

    def _readout(self, x, adjs, node_flags):
        ret = 0.
        for layer, h in enumerate(self.hidden):
            ret = ret + F.dropout(
                self.linear_prediction[layer](h),
                self.dropout_p,
                training=self.training
            )
        return ret.squeeze(-1)  # B x N x F_out

    def _graph_preprocess(self, x, adjs, node_flags):
        adjs = add_self_loop_if_not_exists(adjs)
        # d = adjs.sum(dim=-1)
        # d -= d.min(dim=-1, keepdim=True).values
        # d += 1e-5
        # dh = torch.sqrt(d).reciprocal()
        # adj_hat = dh.unsqueeze(1) * adjs * dh.unsqueeze(-1)
        # adj_hat = torch.softmax(adjs, dim=-1)
        # adj_hat = doubly_stochastic_norm(adjs, do_row_norm=True)
        adj_hat = adjs
        x = (x * node_flags.unsqueeze(-1))

        self.hidden = []
        self.hidden.append(x)
        return x, adj_hat, node_flags

    def get_node_feature(self, x, adjs, node_flags):
        super().get_node_feature(x, adjs, node_flags)
        node_features = torch.cat(self.hidden, dim=-1)
        return node_features
