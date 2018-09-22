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
        # np.random.normal(0, 0.0001, size),
        self.Whx = torch.nn.Parameter(torch.zeros(batch_size, num_hidden))
        self.Whh = torch.nn.Parameter(torch.zeros(num_hidden, num_hidden))
        self.Wph = torch.nn.Parameter(torch.zeros(batch_size, num_hidden))
        self.bh = torch.nn.Parameter(torch.zeros(seq_length))
        self.bp = torch.nn.Parameter(torch.zeros(seq_length))
        self.device = device
        self.h = []
        self.h.append(torch.zeros(batch_size, seq_length))
        self.seq_length = seq_length
        for x in range(seq_length+1):
            self.h.append(torch.zeros(batch_size, num_hidden))
        self.p = []
        for x in range(seq_length):
            self.p.append(torch.zeros(batch_size, input_dim))
        self.softmax = Softmax()
    def forward(self, x):
        for i in range(self.seq_length):
            self.h[i+1] = torch.tanh(torch.mm(self.Whx, x) + torch.mm(self.Whh, self.h[i]) + self.bh)
            self.p[i] = torch.mm(self.Wph, self.h[i+1]) + self.bp
        return self.softmax.forward(self.p[-1])
