import torch
import torch.nn as nn
import torch.nn.functional as F

from model.cond_layers import ConditionalLayer1d


# MLP with linear output

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, use_bn=False, activate_func=F.relu,
                 num_classes=None):
        """
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            num_classes: the number of classes of input, to be treated with different gains and biases,
                    (see the definition of class `ConditionalLayer1d`)
        """

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.use_bn = use_bn
        self.activate_func = activate_func

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            if self.use_bn:
                self.batch_norms = torch.nn.ModuleList()
                for layer in range(num_layers - 1):
                    self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

            self.use_cond = (num_classes is not None)

            if num_classes is not None:
                self.cond_layers = torch.nn.ModuleList()
                for layer in range(num_layers - 1):
                    self.cond_layers.append(ConditionalLayer1d(hidden_dim, num_classes, use_bias=True))

    def forward(self, x):
        """
        :param x: [num_classes * batch_size, N, F_i], batch of node features
            note that in self.cond_layers[layer],
            `x` is splited into `num_classes` groups in dim=0,
            and then treated with different gains and biases
        """
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = self.linears[layer](h)
                if self.use_bn:
                    h = self.batch_norms[layer](h)
                if self.use_cond:
                    h = self.cond_layers[layer](h)
                h = self.activate_func(h)
            return self.linears[self.num_layers - 1](h)
