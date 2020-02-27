import torch
import torch.nn as nn

import torch.nn.functional as F
from model.gnn import GraphNeuralNetwork
from model.pix2pix import UnetSkipConnectionBlockWithResNet
from utils.graph_utils import mask_adjs

# ref: https://github.com/ermongroup/ncsn

class UNetResScore(nn.Module):
    def __init__(self,  nef, max_node_number, dev):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super().__init__()
        # construct unet structure
        input_nc = output_nc = 1
        ngf = nef
        unet_block = UnetSkipConnectionBlockWithResNet(ngf * 8, ngf * 8, input_nc=None, submodule=None, innermost=True)
        unet_block = UnetSkipConnectionBlockWithResNet(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block)
        unet_block = UnetSkipConnectionBlockWithResNet(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block)
        self.model = UnetSkipConnectionBlockWithResNet(output_nc, ngf * 2, input_nc=input_nc, submodule=unet_block,
                                                       outermost=True)  # add the outermost layer
        self.mask = torch.ones([max_node_number, max_node_number]) - torch.eye(max_node_number)
        self.mask.unsqueeze_(0)
        self.mask = self.mask.to(dev)
        self.max_node_num = max_node_number

    def forward(self,  x, adjs, node_flags):
        """Standard forward"""
        adjs = F.interpolate(adjs.unsqueeze(1), size=[32, 32])
        out = self.model(adjs)
        out = F.adaptive_max_pool2d(input=out, output_size=[self.max_node_num, self.max_node_num])
        temp = out.squeeze(1) * self.mask
        return temp + temp.transpose(-1, -2)


class MLPScoreNetwork(nn.Module):
    def __init__(self, nef, max_node_number, dev):
        super().__init__()
        # feature_num_list = [max_node_number**2] + feature_num_list
        # MLPs = []
        edges_num = max_node_number ** 2
        self.net = nn.Sequential(
            nn.Linear(edges_num, edges_num * nef),
            nn.ELU(),
            nn.Linear(edges_num * nef, edges_num * 2 * nef),
            nn.ELU(),
            # nn.BatchNorm1d(edges_num * 4),
            nn.Linear(edges_num * 2 * nef, edges_num * 4 * nef),
            nn.ELU(),
            # nn.BatchNorm1d(edges_num * 8),
            nn.Linear(edges_num * 4 * nef, edges_num * 4 * nef),
            nn.ELU(),
            # nn.BatchNorm1d(edges_num * 8),
            nn.Linear(edges_num * 4 * nef, edges_num * 2 * nef),
            nn.ELU(),
            # nn.BatchNorm1d(edges_num * 8),
            nn.Linear(edges_num * 2 * nef, edges_num * 1 * nef),
            nn.ELU(),
            # nn.BatchNorm1d(edges_num * 8),
            nn.Linear(edges_num * nef, edges_num * 8),
            nn.ELU(),
            # nn.BatchNorm1d(edges_num * 8),
            nn.Linear(edges_num * 8, edges_num * 4),
            nn.ELU(),
            # nn.BatchNorm1d(edges_num * 4),
            nn.Linear(edges_num * 4, edges_num * 2),
            nn.ELU(),
            # nn.BatchNorm1d(edges_num * 2),
            nn.Linear(edges_num * 2, edges_num)
        )
        self.mask = torch.ones([max_node_number, max_node_number]) - torch.eye(max_node_number)
        self.mask.unsqueeze_(0)
        self.mask = self.mask.to(dev)

    def forward(self, x, adjs, node_flags):
        batch_size = adjs.size(0)
        n = adjs.size(-1)
        adjs_vec = adjs.view(batch_size, -1)
        score_vec = self.net(adjs_vec)
        score = score_vec.view(batch_size, -1, n)
        score_s = score + score.transpose(-1, -2)
        return score_s * self.mask


class ConvScore(nn.Module):
    def __init__(self, nef, max_node_number, dev):
        super().__init__()
        nef = nef
        self.conv_net = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(1, nef, 5, stride=1, padding=2),
            # nn.Softplus(),
            # nn.GroupNorm(4, nef),
            nn.ELU(),
            # state size. (nef) x 14 x 14
            nn.Conv2d(nef, nef * 2, 5, stride=1, padding=2),
            # nn.GroupNorm(4, nef * 2),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef*2) x 7 x 7
            nn.Conv2d(nef * 2, nef * 4, 7, stride=1, padding=3),
            # nn.GroupNorm(4, nef * 4),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef*4) x 3 x 3
            nn.ConvTranspose2d(nef * 4, nef * 2, 7, stride=1, padding=3),
            # nn.GroupNorm(4, nef * 2),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef*2) x 7 x 7
            nn.ConvTranspose2d(nef * 2, nef, 5, stride=1, padding=2),
            # nn.GroupNorm(4, nef),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef) x 14 x 14
            nn.ConvTranspose2d(nef, max(1, nef // 2), 5, stride=1, padding=2),
            # nn.Softplus()
            nn.ELU(),

            nn.ConvTranspose2d(max(1, nef // 2), max(1, nef // 4), 3, stride=1, padding=1),
            # nn.Softplus()
            nn.ELU(),

            nn.ConvTranspose2d(max(1, nef // 4), max(1, nef // 8), 1, stride=1, padding=0),
            # nn.Softplus()
            nn.ELU(),

            nn.ConvTranspose2d(max(1, nef // 8), 1, 1, stride=1, padding=0),
            # nn.Softplus()
            # state size. (nc) x 28 x 28
        )
        self.mask = torch.ones([max_node_number, max_node_number]) - torch.eye(max_node_number)
        self.mask.unsqueeze_(0)
        self.mask = self.mask.to(dev)

    def forward(self, x, adjs, node_flags):
        x = adjs
        mask = self.mask.clone()
        mask = mask_adjs(mask, node_flags)
        x = x * mask
        node = x.size(2)
        x = x.view(x.size(0), 1, node, node)
        score = self.conv_net(x).view(x.size(0), node, node)
        score_s = score + score.transpose(-1, -2)
        return score_s * mask


class ScoreNetwork(nn.Module):
    def __init__(self, gnn_module_func, feature_num, stack_num=2):
        super().__init__()
        self.stack_num = stack_num
        self.score_net_list = nn.ModuleList()
        for _ in range(stack_num):
            self.score_net_list.append(OneLayerScoreNetwork(gnn_module_func(), feature_num))

    def forward(self, x, adjs, node_flags):
        for i, score_net in enumerate(self.score_net_list):
            score = score_net(x, adjs, node_flags)
            score = mask_adjs(score, node_flags)
            adjs = adjs + score * 0.1
        return score


class OneLayerScoreNetwork(nn.Module):

    def __init__(self, gnn_module, feature_num):
        super().__init__()
        assert isinstance(gnn_module, GraphNeuralNetwork)
        self.gnn_module = gnn_module
        in_feature = feature_num * 2 + 1
        self.read_score = nn.Sequential(
            nn.Linear(in_feature, feature_num * 4),
            nn.LeakyReLU(),
            nn.Linear(feature_num * 4, feature_num * 4),
            nn.LeakyReLU(),
            nn.Linear(feature_num * 4, feature_num * 4),
            nn.LeakyReLU(),
            nn.Linear(feature_num * 4, feature_num * 2),
            nn.LeakyReLU(),
            nn.Linear(feature_num * 2, 1)
        )
        self.score_gate = nn.Sequential(
            nn.Linear(1, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
            nn.Tanh(),
            nn.Linear(2, 1),
            nn.Tanh()
        )
        self.score_res = nn.Sequential(
            nn.Linear(in_feature, feature_num * 4),
            nn.LeakyReLU(),
            nn.Linear(feature_num * 4, feature_num * 4),
            nn.LeakyReLU(),
            nn.Linear(feature_num * 4, feature_num * 4),
            nn.LeakyReLU(),
            nn.Linear(feature_num * 4, feature_num * 2),
            nn.LeakyReLU(),
            nn.Linear(feature_num * 2, 1)
        )
        # self.add_module('gnn', gnn_module)
        self.add_module('read_score', self.read_score)
        self.add_module('score_gate', self.score_gate)
        self.add_module('score_res', self.score_res)

    def forward(self, x, adjs, node_flags):
        x = self.gnn_module.get_node_feature(x, adjs, node_flags)
        # x = self.read_score(x)  # BS x N x F
        assert isinstance(x, torch.Tensor)
        x_pair = node_feature_to_matrix(x)
        # fake_x_pair = torch.ones_like(x_pair) * adjs.unsqueeze(-1)
        # feed_back_adj = torch.ones_like(adjs)
        feed_back_adj = adjs.unsqueeze(-1)
        score_input = torch.cat([x_pair, feed_back_adj], dim=-1)  # BS x N x N x (2F + 1)
        # score_input = feed_back_adj
        score_1 = self.read_score(score_input).squeeze(-1)  # BS x N x N, pos
        score_gate = self.score_gate(feed_back_adj).squeeze(-1)
        score_res = self.score_res(score_input).squeeze(-1)
        score = score_res * score_gate + score_1
        # score = score_1
        score = score.triu(1) + score.tril(-1)
        ret = score + score.transpose(-1, -2)
        return ret


def node_feature_to_matrix(x):
    """

    :param x:  BS x N x F
    :return:
    x_pair: BS x N x N x 2F
    """
    x_b = x.unsqueeze(-2).expand(-1, -1, x.size(1), -1)  # BS x N x N x F
    x_b_t = x_b.transpose(1, 2)  # BS x N x N x F
    x_pair = torch.cat([x_b, x_b_t], dim=-1)  # BS x N x N x 2F
    return x_pair
