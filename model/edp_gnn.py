import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mlp import MLP
from model.score_network import node_feature_to_matrix
from utils.graph_utils import mask_adjs
from utils.visual_utils import plot_multi_channel_numpy_adjs, plot_multi_channel_numpy_adjs_1b1


class EdgeDensePredictionGNNLayer(nn.Module):
    def __init__(self, gnn_module, c_in, c_out,
                 num_classes=1):
        super().__init__()
        self.multi_channel_gnn_module = gnn_module
        self.translate_mlp = MLP(num_layers=3, input_dim=c_in + 2 * gnn_module.get_out_dim(),
                                 hidden_dim=max(c_in, c_out) * 2, output_dim=c_out,
                                 activate_func=F.elu,
                                 use_bn=True,
                                 num_classes=num_classes)

    def forward(self, x, adjs, node_flags):
        """

        :param x:  B x N x F_i
        :param adjs: B x C_i x N x N
        :param node_flags:  B x N
        :return: x_o: B x N x F_o, new_adjs: B x C_o x N x N
        """
        x_o = self.multi_channel_gnn_module(x, adjs, node_flags)  # B x N x F_o
        x_o_pair = node_feature_to_matrix(x_o)  # B x N x N x 2F_o
        last_c_adjs = adjs.permute(0, 2, 3, 1)  # B x N x N x C_i
        mlp_in = torch.cat([last_c_adjs, x_o_pair], dim=-1)  # B x N x N x (2F_o+C_i)
        mlp_in_shape = mlp_in.shape
        mlp_out = self.translate_mlp(mlp_in.view(-1, mlp_in_shape[-1]))
        new_adjs = mlp_out.view(mlp_in_shape[0], mlp_in_shape[1], mlp_in_shape[2], -1).permute(0, 3, 1, 2)
        new_adjs = new_adjs + new_adjs.transpose(-1, -2)
        # new_adjs = torch.sigmoid(new_adjs)
        new_adjs = mask_adjs(new_adjs, node_flags)
        return x_o, new_adjs


class EdgeDensePredictionGraphScoreNetwork(nn.Module):
    def __init__(self,
                 gnn_module_func,
                 feature_num_list,
                 channel_num_list,
                 max_node_number,
                 dev,
                 gnn_hidden_num_list=(8, 8),
                 num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        gnn_layer_num = len(feature_num_list) - 1
        channel_num_list = [2] + channel_num_list
        assert len(channel_num_list) == len(feature_num_list)
        gnn_hidden_num_list = list(gnn_hidden_num_list)
        self.gnn_list = nn.ModuleList()
        self.final_read_score = MLP(input_dim=sum(channel_num_list), output_dim=1, activate_func=F.elu,
                                    hidden_dim=sum(channel_num_list) * 2,
                                    num_layers=3,
                                    num_classes=num_classes)
        # self.final_read_score = MLP(input_dim=sum(channel_num_list) + sum(feature_num_list)*2,
        #                             output_dim=1, activate_func=torch.tanh,
        #                             hidden_dim=sum(channel_num_list)*2 + sum(feature_num_list)*4,
        #                             num_layers=3)
        for i in range(gnn_layer_num):
            gnn_feature_list = [feature_num_list[i] + channel_num_list[i]] + gnn_hidden_num_list
            gnn = gnn_module_func(feature_nums=gnn_feature_list, out_dim=feature_num_list[i + 1],
                                  channel_num=channel_num_list[i])
            self.gnn_list.append(EdgeDensePredictionGNNLayer(gnn_module=gnn, c_in=channel_num_list[i],
                                                             c_out=channel_num_list[i + 1],
                                                             num_classes=num_classes))

        self.mask = torch.ones([max_node_number, max_node_number]) - torch.eye(max_node_number)
        self.mask.unsqueeze_(0)
        self.mask = self.mask.to(dev)

    def forward(self, x, adjs, node_flags, viz=False, save_dir=None, title=''):
        """
        :param x:  B x N x F_i, batch of node features
        :param adjs: B x C_i x N x N, batch of adjacency matrices
        :param node_flags:  B x N, batch of node_flags, denoting whether a node is effective in each graph
        :param viz: whether to visualize the intermediate channels
        :param title: the filename of the output figure (if viz==True)
        :param save_dir: the directory of the output figure (if viz==True)
        :return: score: B x N x N, the estimated score
        """
        ori_adjs = adjs.unsqueeze(1)
        adjs = torch.cat([ori_adjs, 1. - ori_adjs], dim=1)  # B x 2 x N x N
        adjs = mask_adjs(adjs, node_flags)
        temp_adjs = [adjs]
        # temp_x = [x] if x is not None else []
        for layer in self.gnn_list:
            x, adjs = layer(x, adjs, node_flags)
            temp_adjs.append(adjs)
            # temp_x.append(x)
        if viz:
            batch_size = adjs.size(0) // self.num_classes
            for i in range(self.num_classes):
                plot_multi_channel_numpy_adjs(adjs=[adjs[i * batch_size + 0].detach().cpu().numpy()
                                                    for adjs in temp_adjs],
                                              save_dir=save_dir, title=f's_{i}_' + title)
                plot_multi_channel_numpy_adjs_1b1(
                    adjs=[adjs[i * batch_size + 0].detach().cpu().numpy()
                          for adjs in temp_adjs],
                    save_dir=save_dir, title=f's_{i}_' + title,
                    fig_dir=f'figs_{i}'
                )
                # break
        stacked_adjs = torch.cat(temp_adjs, dim=1)
        # stacked_x = torch.cat(temp_x, dim=-1)  # B x N x sum_F_o
        # stacked_x_pair = node_feature_to_matrix(stacked_x)  # B x N x N x (2sum_F_o)
        mlp_in = stacked_adjs.permute(0, 2, 3, 1)
        # mlp_in = torch.cat([mlp_in, stacked_x_pair], dim=-1)  # B x N x N x (2sum_F_o + sum_C)
        out_shape = mlp_in.shape[:-1]
        mlp_out = self.final_read_score(mlp_in)
        score = mlp_out.view(*out_shape)
        return score * self.mask
