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
from torch.nn import LSTM


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size, lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        self.LSTM = LSTM(vocabulary_size, lstm_num_hidden, num_layers=lstm_num_layers)
        self.fc = nn.Linear(lstm_num_hidden, vocabulary_size)
        self.device = device
        self.batch_size = batch_size
        self.lstm_num_hidden = lstm_num_hidden
        self.seq_length = seq_length
        self.vocabulary_size = vocabulary_size
        self.hidden = self.init_hidden(self.batch_size)

    def forward(self, x):
        x = torch.stack(x)
        list = []
        for i, x_t in enumerate(x):
            x_onehot = torch.FloatTensor(self.batch_size, self.vocabulary_size)
            x_onehot.zero_()
            x_onehot.scatter_(1, x_t.reshape(-1,1), 1)
            list.append(x_onehot)
        x = torch.stack(list)
        x = x.float()
        probs = self.LSTM(x, self.hidden)
        probs = self.fc(probs[0])
        return probs / 0.5

    def init_hidden(self, batch):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(2, batch, self.lstm_num_hidden),
                torch.zeros(2, batch, self.lstm_num_hidden))

    def generate_text(self, x, size):
        self.batch_size = 1
        text = torch.zeros(size)
        text[0] = x.argmax()
        (h,c) = self.init_hidden(1)
        y = 0
        for i in range(size-1):
            x_onehot = torch.FloatTensor(1, self.vocabulary_size)
            x_onehot.zero_()
            x_onehot.scatter_(1, x.reshape(-1, 1), 1)
            x = x_onehot.view(1, 1, self.vocabulary_size)

            probs, (h,c) = self.LSTM(x, (h, c))
            probs = self.fc(probs)
            y = probs.argmax()
            text[i+1] = y
            x = y
        return text
