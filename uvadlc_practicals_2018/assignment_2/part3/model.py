# MIT License
#
# Copyright (c) 2017 Tom Runia
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
from torch.nn import Softmax, Sigmoid


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size, lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        self.Wgx = torch.nn.Parameter(torch.randn(vocabulary_size, lstm_num_hidden))
        self.Wgh = torch.nn.Parameter(torch.randn(lstm_num_hidden, lstm_num_hidden))
        self.Wix = torch.nn.Parameter(torch.randn(vocabulary_size, lstm_num_hidden))
        self.Wih = torch.nn.Parameter(torch.randn(lstm_num_hidden, lstm_num_hidden))
        self.Wfx = torch.nn.Parameter(torch.randn(vocabulary_size, lstm_num_hidden))
        self.Wfh = torch.nn.Parameter(torch.randn(lstm_num_hidden, lstm_num_hidden))
        self.Wox = torch.nn.Parameter(torch.randn(vocabulary_size, lstm_num_hidden))
        self.Woh = torch.nn.Parameter(torch.randn(lstm_num_hidden, lstm_num_hidden))
        self.Wph = torch.nn.Parameter(torch.randn(lstm_num_hidden, vocabulary_size))

        self.bg = torch.nn.Parameter(torch.zeros(lstm_num_hidden))
        self.bi = torch.nn.Parameter(torch.zeros(lstm_num_hidden))
        self.bf = torch.nn.Parameter(torch.zeros(lstm_num_hidden))
        self.bo = torch.nn.Parameter(torch.zeros(lstm_num_hidden))
        self.bp = torch.nn.Parameter(torch.zeros(vocabulary_size))
        self.softmax = Softmax(dim=1)
        self.sigmoid = Sigmoid()
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.lstm_num_hidden = lstm_num_hidden
        self.vocabulary_size = vocabulary_size


    def forward(self, x):
        h = torch.zeros(self.batch_size, self.lstm_num_hidden)
        c = torch.zeros(self.batch_size, self.lstm_num_hidden)
        probs = []
        stacked = torch.stack(x)
        for i in range(self.seq_length):
            x_i = stacked[:, i].reshape(-1,1)
            onehot = torch.FloatTensor(self.batch_size, self.vocabulary_size)
            onehot.zero_()
            onehot.scatter_(1, x_i.reshape(-1, 1), torch.ones(x_i.shape).reshape(-1, 1))

            g_t = torch.tanh(torch.mm(self.Wgx, onehot) + torch.mm(self.Wgh, h) + self.bg)
            i_t = self.sigmoid(torch.mm(self.Wix, onehot) + torch.mm(self.Wih, h) + self.bi)
            f_t = self.sigmoid(torch.mm(self.Wfx, onehot) + torch.mm(self.Wfh, h) + self.bf)
            o_t = self.sigmoid(torch.mm(self.Wox, onehot) + torch.mm(self.Woh, h) + self.bo)
            c_t = g_t * i_t + c * f_t
            h_t = torch.tanh(c_t) * o_t

            c = c_t
            h = h_t
            p_t = torch.mm(h, self.Wph) + self.bp
            y = self.softmax(p_t)
            probs.append(y)
        return probs