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

################################################################################
from torch.nn import Softmax, Sigmoid


class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        self.Wgx = torch.nn.Parameter(torch.randn(num_hidden, input_dim))
        self.Wgh = torch.nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.Wix = torch.nn.Parameter(torch.randn(num_hidden, input_dim))
        self.Wih = torch.nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.Wfx = torch.nn.Parameter(torch.randn(num_hidden, input_dim))
        self.Wfh = torch.nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.Wox = torch.nn.Parameter(torch.randn(num_hidden, input_dim))
        self.Woh = torch.nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.Wph = torch.nn.Parameter(torch.randn(num_classes, num_hidden))

        self.bg = torch.nn.Parameter(torch.zeros([num_hidden,1]))
        self.bf = torch.nn.Parameter(torch.zeros([num_hidden,1]))
        self.bi = torch.nn.Parameter(torch.zeros([num_hidden,1]))
        self.bo = torch.nn.Parameter(torch.zeros([num_hidden,1]))
        self.bp = torch.nn.Parameter(torch.zeros([num_classes,1]))
        self.softmax = Softmax(dim=1)
        self.sigmoid = Sigmoid()
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_hidden = num_hidden

    def forward(self, x):
        h = torch.zeros(self.num_hidden, self.batch_size)
        c = torch.zeros(self.num_hidden, self.batch_size)

        for i in range(self.seq_length):
            x_i = x[:, i].reshape(1,-1)
            g_t = torch.tanh(torch.mm(self.Wgx, x_i) + torch.mm(self.Wgh, h) + self.bg)
            i_t = self.sigmoid(torch.mm(self.Wix, x_i) + torch.mm(self.Wih, h) + self.bi)
            f_t = self.sigmoid(torch.mm(self.Wfx, x_i) + torch.mm(self.Wfh, h) + self.bf)
            o_t = self.sigmoid(torch.mm(self.Wox, x_i) + torch.mm(self.Woh, h) + self.bo)
            c_t = g_t * i_t + c * f_t
            h_t = torch.tanh(c_t) * o_t

            c = c_t
            h = h_t
        p_t = torch.mm(self.Wph, h) + self.bp
        return torch.t(self.softmax(p_t))