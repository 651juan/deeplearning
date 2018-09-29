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

import argparse
import time
from datetime import datetime

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import RMSprop
from torch.utils.data import DataLoader

from part3.dataset import TextDataset
from part3.model import TextGenerationModel
from tensorboardX import SummaryWriter

################################################################################
writer = SummaryWriter('runs_lstm_gen')

def train(config):

    # Initialize the device which to run the model on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size , lstm_num_hidden=config.lstm_num_hidden, lstm_num_layers=config.lstm_num_layers, device=device)

    # Setup the loss and optimizer
    criterion = CrossEntropyLoss()
    optimizer = RMSprop(model.parameters(), lr=config.learning_rate)

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        t1 = time.time()

        batch_targets = torch.stack(batch_targets)
        batch_targets.to(device)
        optimizer.zero_grad()
        print(len(batch_inputs), len(batch_inputs[0]))
        if (len(batch_inputs[0][0]) <64):
            continue
        probs = model.forward(batch_inputs)

        loss = 0
        accuracy = 0
        for prob, target in zip(probs, batch_targets):
            # prediction = torch.argmax(prob, dim=1).float()
            loss += criterion.forward(prob, target)
            predictions = prob.argmax(dim=1).float()
            accuracy += float(torch.sum(predictions == target.float())) / config.batch_size
        loss = loss / config.seq_length
        loss.backward()
        writer.add_scalar('Train/Loss',  loss, step)
        writer.add_scalar('Train/Accurac3y', accuracy, step)
        optimizer.step()
        accuracy = accuracy/ config.seq_length

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % config.print_every == 0:

            print("[{}] Train Step {:04d}/{:04f}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                config.train_steps, config.batch_size, examples_per_second,
                accuracy, loss
            ))

        if step % config.sample_every == 0:
            # Generate some sentences by sampling from the model
            prediction_idx = torch.t(torch.stack([prob.argmax(dim=1) for prob in probs]))
            for b in prediction_idx:
                print("Sentence: ", dataset.convert_to_string(b.numpy()))
                writer.add_text('out', dataset.convert_to_string(b.numpy()))
            pass

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=1000, help='How often to sample from the model')

    config = parser.parse_args()

    # Train the model
    train(config)
