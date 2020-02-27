import torch
import torch.nn as nn

from utils.graph_utils import check_adjs_symmetry

MAX_DEG_FEATURES = 1


class GraphNeuralNetwork(nn.Module):

    def _aggregate(self, x, adjs, node_flags, layer_k):
        """

        :param x: B x N x F_in, the feature vectors of nodes
        :param adjs: B x N x N, the adjacent matrix, with self-loop
        :param node_flags: B x N, the flags for the existence of nodes
        :param layer_k: an int, the index of the layer
        :return: a: B x N x F_mid, the aggregated feature vectors of the neighbors of node
        """
        return x

    def _combine(self, x, a, layer_k):
        """

        :param x: B x N x F_in, the feature vectors of nodes
        :return: a: B x N x F_mid, the aggregated feature vectors of the neighbors of node
        :return: x: B x N x F_out, the feature vectors of nodes
        """
        return a

    @staticmethod
    def _graph_preprocess(x, adjs, node_flags):
        """

        :param x: B x N x F_in, the feature vectors of nodes
        :param adjs: B x N x N, the adjacent matrix, with self-loop
        :param node_flags: B x N, the flags for the existence of nodes
        :return:
            x: B x N x F_in, the feature vectors of nodes
            adjs: B x N x N, the adjacent matrix, with self-loop
            node_flags: B x N, the flags for the existence of nodes
        """
        x = (x * node_flags.unsqueeze(-1))
        check_adjs_symmetry(adjs)
        return x, adjs, node_flags

    @staticmethod
    def _readout(x, adjs, node_flags):
        """

        :param x: B x N x F_in, the feature vectors of nodes
        :param adjs: B x N x N, the adjacent matrix, with self-loop
        :param node_flags: B x N, the flags for the existence of nodes
        :return: energy: B, a float number as the energy of each graph
        """
        x = (x * node_flags.unsqueeze(-1))
        return x.view(x.size(0), -1).sum(-1).squeeze()

    def __init__(self, max_layers_num):
        super().__init__()
        self.max_layers_num = max_layers_num

    def get_node_feature(self, x, adjs, node_flags):
        deg = adjs.sum(-1).unsqueeze(-1)  # B x C x N x 1 or B x N x 1
        if len(deg.shape) == 4:
            deg = deg.permute(0, 2, 1, 3).contiguous().view(adjs.size(0), adjs.size(-1), -1)
        if x is None:
            x = deg
        else:
            x = torch.cat([x, deg], dim=-1)
        x, adjs, node_flags = self._graph_preprocess(x, adjs, node_flags)
        for k in range(self.max_layers_num):
            x = self._combine(x=x, a=self._aggregate(x=x, adjs=adjs, node_flags=node_flags, layer_k=k), layer_k=k)
        return x

    def forward(self, x, adjs, node_flags):
        x = self.get_node_feature(x, adjs, node_flags)
        return self._readout(x, adjs, node_flags)  # energy for each graph
