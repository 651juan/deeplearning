"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features, learning_rate=2e-3):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """
    size = ((in_features,) + (out_features,))
    self.params = {'weight': np.random.normal(0, 0.0001, size), 'bias': np.zeros(out_features)}
    self.grads = {'weight': np.zeros(size), 'bias': np.zeros(out_features)}
    self.inter = {}
    self.LEARNING_RATE = learning_rate

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    self.inter['previousX'] = x
    out = x.dot(self.params['weight']) + self.params['bias']

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """
    dx = dout.dot(self.params['weight'].T)
    self.grads['weight'] = (self.inter['previousX'].T).dot(dout)
    self.grads['bias'] = np.sum(dout, axis=0)
    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def __init__(self):
    self.inter = {}

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    self.inter['previousXPos'] = x > 0
    out = x * self.inter['previousXPos']

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """
    dx = self.inter['previousXPos'].astype(int)
    return dx * dout

class SoftMaxModule(object):
  def __init__(self):
    self.inter = {}
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    self.inter['previousX'] = x
    b = x.max(axis=0,keepdims=True)
    y = np.exp(x - b)
    out = y / y.sum(axis=0,keepdims=True)
    self.inter['probs'] = out
    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:outout
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """
    probs = self.inter['probs'].reshape(-1, 1)
    jacobian = np.diagflat(probs) - np.dot(probs, probs.T)
    shape = np.shape(self.inter['probs'])
    return dout.reshape(1,-1).dot(jacobian).reshape(shape[0], shape[1])
class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    
    TODO:
    Implement forward pass of the module. 
    """
    yIsTrue = np.where(y==1)
    out = - np.sum(np.log(x[yIsTrue])) / y.shape[0]
    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    
    TODO:
    Implement backward pass of the module.
    """
    out = -y/(x*y.shape[0]*y)
    positions = np.isnan(out)
    out[positions] = 0
    return out
