from torch import FloatTensor
import numpy as np


class Parameter(object):
    """
    Container for the Module parameters, contains value and gradient.
    """
    def __init__(self, value=[], gradient=[]):
        self.value = value
        self.gradient = gradient


class Module(object):
    pass


class Linear(Module):
    """
    Linear/fully connected module.

    Parameters
    ----------
    dim_in : int
        Dimention of the input.
    dim_out : int
        Dimention in the ouput.
    """
    def __init__(self, dim_in, dim_out, std=0.05):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.w = Parameter(FloatTensor(std*np.random.randn(dim_in, dim_out)))
        self.b = Parameter(FloatTensor(np.zeros(dim_out)))

    def forward(self, input):
        assert input.dim() == 2
        self.input = input
        return input @ self.w.value + self.b.value

    def backward(self, gradwrtoutput):
        assert gradwrtoutput.dim() == 2
        self.w.gradient = gradwrtoutput.t() @ self.input
        self.b.gradient = gradwrtoutput.sum(0)
        return gradwrtoutput @ self.w.value.t()

    def get_param(self):
        return (self.w, self.b)
