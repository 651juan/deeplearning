"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn


class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """

    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem


        TODO:
        Implement initialization of the network.
        """
        super(ConvNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(3, 64, (3, 3), 1, 1))
        self.layers.append(nn.BatchNorm2d(64))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d((3, 3), 2, 1))
        self.layers.append(nn.Conv2d(64, 128, (3, 3), 1, 1))
        self.layers.append(nn.BatchNorm2d(128))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d((3, 3), 2, 1))
        self.layers.append(nn.Conv2d(128, 256, (3, 3), 1, 1))
        self.layers.append(nn.BatchNorm2d(256))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(256, 256, (3, 3), 1, 1))
        self.layers.append(nn.BatchNorm2d(256))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d((3, 3), 2, 1))
        self.layers.append(nn.Conv2d(256, 512, (3, 3), 1, 1))
        self.layers.append(nn.BatchNorm2d(512))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(512, 512, (3, 3), 1, 1))
        self.layers.append(nn.BatchNorm2d(512))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d((3, 3), 2, 1))
        self.layers.append(nn.Conv2d(512, 512, (3, 3), 1, 1))
        self.layers.append(nn.BatchNorm2d(512))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(512, 512, (3, 3), 1, 1))
        self.layers.append(nn.BatchNorm2d(512))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d((3, 3), 2, 1))
        self.layers.append(nn.AvgPool2d((1, 1), 1, 0))
        self.layers.append(nn.Linear(512, 10))

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        out = x
        for i, l in enumerate(self.layers):
            if isinstance(self.layers[i], nn.Linear):
                out = out.squeeze()
            out = self.layers[i](out)
        return out
