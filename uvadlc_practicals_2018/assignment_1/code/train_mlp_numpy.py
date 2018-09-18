"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule, SoftMaxModule
import cifar10_utils

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 100
EVAL_FREQ_DEFAULT = 100
from tensorboardX import SummaryWriter

writer = SummaryWriter('runs_mlp')

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
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  # Get Images
  cifar10 = cifar10_utils.read_data_sets(DATA_DIR_DEFAULT)
  # Create MLP Instance
  trainDataSet = cifar10['train']
  testDataSet = cifar10['test']
  size_of_images =  cifar10['train'].images[0].shape[0] * cifar10['train'].images[0].shape[1] * cifar10['train'].images[0].shape[2]
  mlp = MLP(size_of_images, dnn_hidden_units, np.shape(cifar10['test'].labels)[1])
  loss = CrossEntropyModule()
  aggregate_counter = 0
  for i in range(FLAGS.max_steps):
    # np.random.shuffle(cifar10['train'])
    accuracies_train = []
    loss_train = []
    flag = trainDataSet.epochs_completed
    counter = 0
    while flag == trainDataSet.epochs_completed:
      counter = counter + 1
      batch = trainDataSet.next_batch(BATCH_SIZE_DEFAULT)
      x = batch[0]
      x = x.reshape(x.shape[0], (x.shape[1]*x.shape[2]*x.shape[3]))
      y = batch[1]

      prob = mlp.forward(x)
      predictions = (prob == prob.max(axis=1)[:, None]).astype(int)
      current_accuracy = accuracy(predictions, y)
      accuracies_train.append(current_accuracy)
      current_loss = loss.forward(prob, y)
      loss_train.append(current_loss)
      niter = aggregate_counter + counter
      out_loss_back = loss.backward(prob,y)
      mlp.backward(out_loss_back)
      writer.add_scalar('Train/Loss', current_loss, niter)
      writer.add_scalar('Train/Accuracy', current_accuracy, niter)
    if i % FLAGS.eval_freq == 0:
      test_dataset(mlp, testDataSet, loss, aggregate_counter, i)
    aggregate_counter += counter
    writer.add_scalar('Train/LossIteration', np.mean(loss_train), i)
    writer.add_scalar('Train/AccuracyIteration', np.mean(accuracies_train), i)
    print(np.mean(accuracies_train))
    print(np.mean(loss_train))

def test_dataset(mlp, testDataSet, loss, agg, i):
  accuracies_test = []
  loss_test = []
  flag = testDataSet.epochs_completed
  counter = 0
  while flag == testDataSet.epochs_completed:
    counter = counter + 1
    batch = testDataSet.next_batch(BATCH_SIZE_DEFAULT)
    x = batch[0]
    x = x.reshape(x.shape[0], (x.shape[1] * x.shape[2] * x.shape[3]))
    y = batch[1]
    prob = mlp.forward(x)
    predictions = (prob == prob.max(axis=1)[:, None]).astype(int)
    current_accuracy = accuracy(predictions, y)
    accuracies_test.append(current_accuracy)
    current_loss = loss.forward(prob, y)
    loss_test.append(current_loss)
    niter = agg + counter
    writer.add_scalar('Test/Loss', current_loss, niter)
    writer.add_scalar('Test/Accuracy', current_accuracy, niter)
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
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
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