################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np

################################################################################
from torch.nn import Softmax


class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        gaussian = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([0.1]))

        self.Whx = torch.nn.Parameter(torch.randn(input_dim, num_hidden))
        self.Whh = torch.nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.Wph = torch.nn.Parameter(torch.randn(batch_size, num_classes))

        self.bh = torch.nn.Parameter(torch.zeros(num_hidden))
        self.bp = torch.nn.Parameter(torch.zeros(num_classes))

        self.device = device
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_hidden = num_hidden
        self.softmax = Softmax(dim=1)

    def forward(self, x):
        h = torch.zeros(self.batch_size, self.num_hidden)
        for i in range(self.seq_length):
            x_i = x[:, i].reshape(-1,1)
            h_t = torch.tanh(torch.mm(self.Whx, x_i) + torch.mm(self.Whh, h) + self.bh)
            h = h_t
            p = torch.mm(h_t, self.Wph) + self.bp
        return self.softmax(p)