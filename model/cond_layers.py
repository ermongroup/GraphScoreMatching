import torch
from torch import nn as nn


class ConditionalLayer1d(nn.Module):
    def __init__(self, num_features, num_classes, use_bias=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.use_bias = use_bias
        self.gain = nn.Parameter(torch.randn(num_classes, 1, num_features) * 0.02 + 1.)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(num_classes, 1, num_features))
        else:
            self.bias = None

    def forward(self, x):
        """

        :param x: (num_classes x BS x N) x F or (num_classes x BS x N^2) x F
        :return: x_out
        """
        x_class_grouped = x.view(self.num_classes, -1, self.num_features)
        # num_classes x (BS x N) x F or num_classes x (BS x N^2) x F
        x_out = x_class_grouped * self.gain
        if self.use_bias:
            x_out = x_out + self.bias
        return x_out.view(-1, self.num_features)  # (num_classes x BS x N) x F or (num_classes x BS x N^2) x F

