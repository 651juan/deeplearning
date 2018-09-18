import numpy as np
import torch
import torch.nn as nn

"""
The modules/function here implement custom versions of batch normalization in PyTorch.
In contrast to more advanced implementations no use of a running mean/variance is made.
You should fill in code into indicated sections.
"""

######################################################################################
# Code for Question 3.1
######################################################################################

class CustomBatchNormAutograd(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  The operations called in self.forward track the history if the input tensors have the
  flag requires_grad set to True. The backward pass does not need to be implemented, it
  is dealt with by the automatic differentiation provided by PyTorch.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormAutograd object. 
    
    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability
    
    TODO:
      Save parameters for the number of neurons and eps.
      Initialize parameters gamma and beta via nn.Parameter
    """
    super(CustomBatchNormAutograd, self).__init__()
    self.n_neurons = n_neurons
    self.eps = eps
    self.beta = nn.Parameter(torch.zeros(n_neurons))
    self.gamma = nn.Parameter(torch.ones(n_neurons))

  def forward(self, input):
    """
    Compute the batch normalization
    
    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor
    
    TODO:
      Check for the correctness of the shape of the input tensor.
      Implement batch normalization forward pass as given in the assignment.
      For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
    """
    for name, param in self.named_parameters():
      print(name)
    if (input.size()[1] != self.n_neurons):
      raise ValueError("Dimensions mismatch")

    mu = torch.mean(input, 0)
    theta = torch.var(torch.add(input,-mu),0, unbiased=False) #torch.mean((input - mu) ** 2, 0)
    norm = torch.div((torch.add(input,-mu)), torch.sqrt(torch.add(theta,self.eps)))
    out = torch.add(torch.mul(self.gamma, norm), self.beta)
    return out



######################################################################################
# Code for Question 3.2 b)
######################################################################################


class CustomBatchNormManualFunction(torch.autograd.Function):
  """
  This torch.autograd.Function implements a functional custom version of the batch norm operation for MLPs.
  Using torch.autograd.Function allows you to write a custom backward function.
  The function will be called from the nn.Module CustomBatchNormManualModule
  Inside forward the tensors are (automatically) not recorded for automatic differentiation since the backward
  pass is done via the backward method.
  The forward pass is not called directly but via the apply() method. This makes sure that the context objects
  are dealt with correctly. Example:
    my_bn_fct = CustomBatchNormManualFunction()
    normalized = fct.apply(input, gamma, beta, eps)
  """

  @staticmethod
  def forward(ctx, input, gamma, beta, eps=1e-5):
    """
    Compute the batch normalization
    
    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
      input: input tensor of shape (n_batch, n_neurons)
      gamma: variance scaling tensor, applied per neuron, shpae (n_neurons)
      beta: mean bias tensor, applied per neuron, shpae (n_neurons)
      eps: small float added to the variance for stability
    Returns:
      out: batch-normalized tensor

    TODO:
      Implement the forward pass of batch normalization
      Store constant non-tensor objects via ctx.constant=myconstant
      Store tensors which you need in the backward pass via ctx.save_for_backward(tensor1, tensor2, ...)
      Intermediate results can be decided to be either recomputed in the backward pass or to be stored
      for the backward pass. Do not store tensors which are unnecessary for the backward pass to save memory!
      For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
    """
    N, D = input.shape

    # step1: calculate mean
    mu = 1. / N * torch.sum(input, 0)

    # step2: subtract mean vector of every trainings example
    xmu = input - mu

    # step3: following the lower branch - calculation denominator
    sq = xmu ** 2

    # step4: calculate variance
    var = 1. / N * torch.sum(sq, 0)

    # step5: add eps for numerical stability, then sqrt
    sqrtvar = np.sqrt(var + eps)

    # step6: invert sqrtwar
    ivar = 1. / sqrtvar

    # step7: execute normalization
    xhat = xmu * ivar

    # step8: Nor the two transformation steps
    gammax = gamma * xhat

    # step9
    out = gammax + beta

    # store intermediate
    ctx.xhat = xhat
    ctx.gamma = gamma
    ctx.xmu = xmu
    ctx.ivar = ivar
    ctx.sqrtvar = sqrtvar
    ctx.var = var
    ctx.eps = eps

    return out


  @staticmethod
  def backward(ctx, dout):
    """
    Compute backward pass of the batch normalization.
    
    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
    Returns:
      out: tuple containing gradients for all input arguments
    
    TODO:
      Retrieve saved tensors and constants via ctx.saved_tensors and ctx.constant
      Compute gradients for inputs where ctx.needs_input_grad[idx] is True. Set gradients for other
      inputs to None. This should be decided dynamically.
    """

    N, D = dout.shape

    # step9
    dbeta = torch.sum(dout, 0)
    dgammax = dout  # not necessary, but more understandable

    # step8
    dgamma = torch.sum(dgammax * ctx.xhat,0)
    dxhat = dgammax * ctx.gamma

    # step7
    divar = torch.sum(dxhat * ctx.xmu, 0)
    dxmu1 = dxhat * ctx.ivar

    # step6
    dsqrtvar = -1. / (ctx.sqrtvar ** 2) * divar

    # step5
    dvar = 0.5 * 1. / np.sqrt(ctx.var + ctx.eps) * dsqrtvar

    # step4
    dsq = 1. / N * np.ones((N, D)) * dvar

    # step3
    dxmu2 = 2 * ctx.xmu * dsq

    # step2
    dx1 = (dxmu1 + dxmu2)
    dmu = -1 * torch.sum(dxmu1 + dxmu2, 0)

    # step1
    dx2 = 1. / N * np.ones((N, D)) * dmu

    # step0
    dx = dx1 + dx2

    # return gradients of the three tensor inputs and None for the constant eps
    return dx, dgamma, dbeta, None



######################################################################################
# Code for Question 3.2 c)
######################################################################################

class CustomBatchNormManualModule(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  In self.forward the functional version CustomBatchNormManualFunction.forward is called.
  The automatic differentiation of PyTorch calls the backward method of this function in the backward pass.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormManualModule object.
    
    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability
    
    TODO:
      Save parameters for the number of neurons and eps.
      Initialize parameters gamma and beta via nn.Parameter
    """
    super(CustomBatchNormManualModule, self).__init__()
    self.n_neurons = n_neurons
    self.eps = eps
    self.beta = nn.Parameter(torch.zeros(n_neurons))
    self.gamma = nn.Parameter(torch.ones(n_neurons))

  def forward(self, input):
    """
    Compute the batch normalization via CustomBatchNormManualFunction
    
    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor
    
    TODO:
      Check for the correctness of the shape of the input tensor.
      Instantiate a CustomBatchNormManualFunction.
      Call it via its .apply() method.
    """
    if (input.size()[1] != self.n_neurons):
      raise ValueError("Dimensions mismatch")
    customBatch = CustomBatchNormManualFunction()
    out = customBatch.apply(input, self.gamma, self.beta)

    return out
