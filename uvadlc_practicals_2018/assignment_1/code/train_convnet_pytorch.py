"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import torch
from tensorboardX import SummaryWriter

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter('runs_convnet')

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  accuracy = np.sum(np.all((predictions == targets), axis=1)) / predictions.shape[0]

  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)
  print(device)
  # Get Images
  cifar10 = cifar10_utils.read_data_sets(FLAGS.data_dir)
  # Create MLP Instance
  trainDataSet = cifar10['train']
  testDataSet = cifar10['test']

  mlp = ConvNet(cifar10['train'].images[0].shape[2], np.shape(cifar10['test'].labels)[1]).to(device)
  loss = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(mlp.parameters(), lr=FLAGS.learning_rate)
  aggregate_counter = 0
  for i in range(FLAGS.max_steps):
    # np.random.shuffle(cifar10['train'])
    accuracies_train = []
    loss_train = []
    counter = 0
    batch = trainDataSet.next_batch(FLAGS.batch_size)
    x = torch.from_numpy(batch[0]).to(device)
    # x = torch.from_numpy(x.reshape(x.shape[0], x.shape[2], x.shape[3], x.shape[1])).to(device)
    y_numpy = batch[1]
    y = torch.from_numpy(batch[1]).to(device)
    optimizer.zero_grad()
    prob = mlp(x)
    prob_num = prob.cpu().clone().detach().numpy()
    predictions = (prob_num == prob_num.max(axis=1)[:, None]).astype(int)
    current_accuracy = accuracy(predictions, y_numpy)
    print(current_accuracy)
    accuracies_train.append(current_accuracy)

    current_loss = loss(prob, torch.max(y, 1)[1])
    current_loss.backward()
    optimizer.step()
    current_loss = current_loss.cpu().detach().numpy()
    loss_train.append(current_loss)
    writer.add_scalar('Train/Loss', current_loss, i)
    writer.add_scalar('Train/Accuracy', current_accuracy, i)
    if i % FLAGS.eval_freq == 0:
        test_dataset(mlp, testDataSet, loss, i)
    aggregate_counter += counter
    print("ITERATION FINISHED", i, " ", np.mean(accuracies_train))
  test_dataset(mlp, testDataSet, loss, FLAGS.max_steps +1)


def test_dataset(mlp, testDataSet, loss, i):
    accuracies_test = []
    loss_test = []
    with torch.no_grad():
        flag = testDataSet.epochs_completed
        counter = 0
        while flag == testDataSet.epochs_completed:
            counter = counter + 1
            batch = testDataSet.next_batch(FLAGS.batch_size)
            x = torch.from_numpy(batch[0]).to(device)
            # x = torch.from_numpy(x.reshape(x.shape[0], (x.shape[1] * x.shape[2] * x.shape[3]))).to(device)
            y_numpy = batch[1]
            y = torch.from_numpy(batch[1]).to(device)
            outputs = mlp(x)
            outputs_num = outputs.cpu().detach().numpy()
            predictions = (outputs_num == outputs_num.max(axis=1)[:, None]).astype(int)
            current_accuracy = accuracy(predictions, y_numpy)
            accuracies_test.append(current_accuracy)
            current_loss = loss(outputs, torch.max(y, 1)[1])
            current_loss = current_loss.cpu().detach().numpy()
            loss_test.append(current_loss)
        writer.add_scalar('Test/LossIteration', np.mean(loss_test), i)
        writer.add_scalar('Test/AccuracyIteration', np.mean(accuracies_test), i)

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()