from torch import FloatTensor, tanh, max, sum
import torch
import numpy as np

# FIXME The code is written for batch processing but the way the tensor
# dimentions work, it will break for batch_size = 1


class Parameter(object):
    """
    Container for the Module parameters, contains value and gradient.
    """
    def __init__(self, value=[], gradient=[]):
        self.value = value
        self.gradient = gradient


class Module(object):
    pass


class Optimizer(object):
    pass


class Loss(object):
    pass


class SGD(Optimizer):
    """
    Gradient descent optimizer.

    Parameters
    ----------
    eta : float
        Step size.
    """
    def __init__(self, eta=1e-3):
        self.eta = eta

    def apply(self, layers):
        for layer in layers:
            param = layer.get_param()
            for p in param:
                # print("param = ", p.value)
                # print("dparam = ", self.eta * p.gradient)
                p.value -= self.eta * p.gradient


class Adam(Optimizer):
    """
    Adam optimzer.
    """
    pass


class MSE(Loss):
    def loss(self, output, labels):
        return sum((output - labels)**2)/labels.shape[1]

    def dloss(self, output, labels):
        """
        Derivative of the loss as function of ouput of the network.
        """
        # print("output = ", output, ", labels = ", labels)
        # print("dloss = ", 2*(output - labels)/labels.shape[1])
        return 2*(output - labels)/labels.shape[1]


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


class ReLU(Module):
    """
    Rectified Linear Unit module. Performes the operation max(0, x) on the
    input tensor in an element-wise manner.
    """
    def forward(self, input):
        assert input.dim() == 2
        self.input = input
        return max(input, FloatTensor([0]))

    def backward(self, gradwrtoutput):
        assert gradwrtoutput.dim() == 2
        return (self.input > 0).float() * gradwrtoutput

    def get_param(self):
        return []


class Tanh(Module):
    """
    Hyperbolic tangeant module. Performes the tanh(x) operation on the input
    tennor in element-wise manner.
    """
    def forward(self, input):
        assert input.dim() == 2
        self.input = input
        return tanh(input)

    def backward(self, gradwrtoutput):
        assert gradwrtoutput.dim() == 2
        return (1 - tanh(self.input)**2) * gradwrtoutput

    def get_param(self):
        return []


class Sigmoid(Module):
    """
    Hyperbolic tangeant module. Performes the tanh(x) operation on the input
    tennor in element-wise manner.
    """
    def forward(self, input):
        assert input.dim() == 2
        self.input = input
        self.output = torch.sigmoid(input)
        return self.output

    def backward(self, gradwrtoutput):
        assert gradwrtoutput.dim() == 2
        # print("ds = ", self.output * (1 - self.output) * gradwrtoutput)
        return self.output * (1 - self.output) * gradwrtoutput

    def get_param(self):
        return []


class Sequential(object):
    """
    Creates a network from a model from a list a modules.

    Parameters
    ----------
    layers : list
        A list of the different modules of the model. The dimention of the
        output of each layer must match the dimention of the input of the next
        layer.
    """
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        self.output = x
        return self.output

    def backward(self, gradwrtoutput):
        y = gradwrtoutput
        for l in reversed(self.layers):
            y = l.backward(y)

    def get_layers(self):
        return self.layers

    def train(self, x, y, optimizer, loss, batch_size=1, epochs=1,
              verbose=True):

        # Casting into 2D tensor
        if y.dim() == 1:
            y = y.unsqueeze(1)

        # Extract batches
        batches_x = x.split(batch_size, dim=0)
        batches_y = y.split(batch_size, dim=0)

        # Training
        for e in range(epochs):
            loss_val = 0
            accuracy = 0
            if verbose:
                print("Epoch {0:d}/{1:d}".format(e+1, epochs))

            for batch_x, batch_y in zip(batches_x, batches_y):
                # Forward Pass
                output = self.forward(batch_x)
                grad = loss.dloss(output, batch_y)

                # Backward Pass
                self.backward(grad)
                optimizer.apply(self.layers)

                # Update tracking metric
                if verbose:
                    loss_val += loss.loss(output, batch_y)

            # Display message
            if verbose:
                print("Loss = {0:.2e}".format(loss_val, accuracy*100))

    def predict(self, x, batch_size=None):
        # Prep Ouput
        output = FloatTensor()

        # Casting into 2D tensor if required
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Extracting batches
        if batch_size is None:
            batches = [x]
        else:
            batches = x.split(batch_size, dim=0)

        # Propagating batches
        for batch in batches:
            # Forward Pass
            output = torch.cat((output, self.forward(batch)), dim=0)

        return output
