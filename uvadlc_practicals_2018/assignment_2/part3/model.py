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
from torch.nn import LSTM, Softmax
import numpy as np

class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size, lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        self.LSTM = LSTM(1, lstm_num_hidden, num_layers=lstm_num_layers, batch_first=True)
        self.fc = nn.Linear(lstm_num_hidden, vocabulary_size)
        self.softmax = Softmax(dim=2)
        self.device = device

    def forward(self, x):
        tensor = torch.stack(x)
        tensor = torch.unsqueeze(tensor,2).float()
        tensor.to(self.device)
        probs = self.LSTM(tensor)
        probs = self.fc(probs[0])
        return self.softmax(probs)