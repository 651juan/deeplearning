"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
    
    TODO:
    Implement initialization of the network.
    """
    self.layers = []
    previous_size = n_inputs

    for layer in range(len(n_hidden)):
        linear = LinearModule(previous_size, n_hidden[layer])
        relu = ReLUModule()
        previous_size = n_hidden[layer]
        self.layers.append(linear)
        self.layers.append(relu)

    linear = LinearModule(previous_size, n_classes)
    self.layers.append(linear)
    softmax = SoftMaxModule()
    self.layers.append(softmax)

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
    for layer in self.layers:
        out = layer.forward(out)
    return out

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    
    TODO:
    Implement backward pass of the network.
    """
    dx = dout
    for layer in reversed(self.layers):
        dx = layer.backward(dx)
        if (isinstance(layer, LinearModule)):
            layer.params['weight'] -= layer.LEARNING_RATE * layer.grads['weight']
            layer.params['bias'] -= layer.LEARNING_RATE * layer.grads['bias']
    return
