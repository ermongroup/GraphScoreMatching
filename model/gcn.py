import torch
import torch.nn.functional as F

from model.gcn_layer import GraphConvolution
from model.gnn import GraphNeuralNetwork
from utils.graph_utils import add_self_loop_if_not_exists


class GCN(GraphNeuralNetwork):

    def __init__(self, feature_nums, dropout_p=0.5, **kwargs):
        layer_n = len(feature_nums) - 1
        super().__init__(layer_n)
        layers = []
        for i in range(layer_n):
            gcn = GraphConvolution(feature_nums[i], feature_nums[i + 1])
            layers.append(gcn)
            self.add_module(name=f'gcn_{i}', module=gcn)
        self.layers = layers
        self.dropout_p = dropout_p

    def _aggregate(self, x, adjs, node_flags, layer_k):
        a = self.layers[layer_k](x, adjs)
        # a = F.dropout(a, self.dropout_p, training=self.training)
        if layer_k < self.max_layers_num - 1:
            a = F.relu(a)
        # a = F.relu(a)
        return a

    def _combine(self, x, a, layer_k):
        return a

    @staticmethod
    def _graph_preprocess(x, adjs, node_flags):
        adjs = add_self_loop_if_not_exists(adjs)
        d = adjs.sum(dim=-1)
        dh = torch.sqrt(d).reciprocal()
        adj_hat = dh.unsqueeze(1) * adjs * dh.unsqueeze(-1)
        x = (x * node_flags.unsqueeze(-1))
        return x, adj_hat, node_flags
