import torch
import collections
import math
from torch import FloatTensor, tanh, normal, zeros, ones, sigmoid, cat, sqrt,\
    arange, max, sum

###############################################################################
#   optimizers.py
###############################################################################


class Optimizer(object):
    def apply(self):
        raise NotImplementedError


class SGD(Optimizer):
    """Gradient descent optimizer.

    Parameters
    ----------
    lr : float
        Learning rate (i.e. step size).
    lamb : float
        Momentum.
    """
    def __init__(self, lr=1e-2, lamb=0.5):
        self.lr = lr
        self.lamb = lamb

    def apply(self, layers):
        for layer in layers:
            param = layer.get_param()
            for p in param:
                p.m = self.lamb*p.m + self.lr*p.gradient
                p.value -= p.m


class Adam(Optimizer):
    """Adam optimzer.

    Parameters
    ----------
    lr : float
        Learning rate (i.e. step size).
    rho1 : float
    rho2 : float
    delta : float
    """
    def __init__(self, lr=1e-3, rho1=0.9, rho2=0.999, delta=1e-8):
        self.lr = lr
        self.rho1 = rho1
        self.rho2 = rho2
        self.delta = delta
        self.t = 0

    def apply(self, layers):
        self.t = self.t + 1
        for layer in layers:
            param = layer.get_param()
            for p in param:
                new_m = self.rho1*p.m + (1-self.rho1)*p.gradient
                new_v = self.rho2*p.v + (1-self.rho2)*p.gradient**2
                p.m = new_m
                p.v = new_v

                m_hat = p.m/(1-self.rho1**self.t)
                v_hat = p.v/(1-self.rho2**self.t)

                p.value -= self.lr*m_hat/(sqrt(v_hat)+self.delta)


###############################################################################
#   activations.py
###############################################################################


class Activations(object):
    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def get_param(self):
        raise NotImplementedError


class ReLU(Activations):
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


class Tanh(Activations):
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


class Sigmoid(Activations):
    """
    Hyperbolic tangeant module. Performes the tanh(x) operation on the input
    tennor in element-wise manner.
    """
    def forward(self, input):
        assert input.dim() == 2
        self.input = input
        self.output = sigmoid(input)
        return self.output

    def backward(self, gradwrtoutput):
        assert gradwrtoutput.dim() == 2
        # print("ds = ", self.output * (1 - self.output) * gradwrtoutput)
        return self.output * (1 - self.output) * gradwrtoutput

    def get_param(self):
        return []


###############################################################################
#   layers.py
###############################################################################


class Parameter(object):
    """
    Container for the Module parameters, contains value and gradient.
    If ADAM : m,u standard parameters
    if SGD :  the m parameters is used for the momentum
    """
    def __init__(self, value=[], gradient=[]):
        self.value = value
        self.gradient = gradient
        self.m = 0
        self.v = 0


class Module(object):
    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def get_param(self):
        raise NotImplementedError


class Linear(Module):
    """Linear/fully connected module.

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
        self.w = zeros(dim_in, dim_out)
        self.w = Parameter(normal(zeros(dim_in, dim_out),
                                  std * ones(dim_in, dim_out)))
        self.b = Parameter(zeros(dim_out))

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

###############################################################################
#   models.py
###############################################################################


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
        """Trains the model over a set number of epochs using a specified
        training data, optimizer and loss function and batch size.

        Parameters
        ----------
        x: FloatTensor
            Input training data.
        y: FloatTensor
            Traing labels.
        optmizer: Optimizer object
            This an instance of the optizer that will be used to train the
            model.
        loss: Loss object
            The is an instance of the loss function that will be used to train
            the model.
        batch_size: int
            Batch size.
        epochs: int
            Number of epochs to train over.
        """

        nb_samples = x.shape[0]

        # Casting into 2D tensor
        if y.dim() == 1:
            y = y.unsqueeze(1)

        # Extract batches
        batches_x = x.split(batch_size, dim=0)
        batches_y = y.split(batch_size, dim=0)

        # Training
        for e in range(epochs):
            total_loss = 0
            accuracy = 0
            if verbose:
                print("\nEpoch {0:d}/{1:d}".format(e+1, epochs))

            for batch_x, batch_y in zip(batches_x, batches_y):
                # Forward Pass
                output = self.forward(batch_x)
                grad = loss.dloss(output, batch_y)

                # Backward Pass
                self.backward(grad)
                optimizer.apply(self.layers)

                # Update tracking metric
                if verbose:
                    total_loss += sum(loss.loss(output, batch_y))

            # Display message
            if verbose:
                average_loss = total_loss/nb_samples
                print("Loss = {0:.2e}".format(average_loss, accuracy*100))

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
            output = cat((output, self.forward(batch)), dim=0)

        return output.squeeze()


###############################################################################
#   losses.py
###############################################################################


class Loss(object):
    def loss(self):
        raise NotImplementedError

    def dloss(self):
        raise NotImplementedError


class MSE(Loss):
    """Mean Square Error (MSE) loss function.
    """
    def loss(self, output, labels):
        """Value of loss.

        Parameters
        ----------
        output : Tensor
            Output of the network. This a Tensor of shape
            (nb_samples, output_dim).
        labels : Tensor
            The true labels. This is a Tensor of the same shape as 'output'.

        Returns
        -------
        The MSE loss between the 'output' ans the true 'labels'. This is a
        Tensor of shape (nb_samples, 1).
        """
        return sum((output - labels)**2, dim=1)/labels.size()[1]

    def dloss(self, output, labels):
        """Derivative of the loss as function of ouput of the network.

        Parameters
        ----------
        output : Tensor
            Output of the network. This a Tensor of shape
            (nb_samples, output_dim).
        labels : Tensor
            The true labels. This is a Tensor of the same shape as 'output'.

        Returns
        -------
        The derivative of the loss with respect to the input. This a Tensor of
        the same shape as 'output'.
        """
        return 2*(output - labels)/labels.size()[1]


###############################################################################
#   utilities.py
###############################################################################


def to_onehot(labels, nb_classes):
    """ Converts integer labels to one-hot encoding.

    Parameters
    ----------
    labels: int vector
        Vector cotaining the labels to convert.
    nb_classes: int
        Number of possible classes.

    Returns
    -------
    A tensor of size (nb_samples, nb_classes) containing the one-hot labels.
    """
    labels_onehot = labels.view(-1, 1) == arange(nb_classes).view(1, -1)
    return labels_onehot.float()


def getAccuracy(true_labels, pred_labels, one_hot=False):
    """Calculates accuracy based on true labels and labels predicted by a
    model.

    Parameters
    ----------
    true_labels: int Tensor
        Tensor of true labels, 2d if one-hot encoding is used, 1d otherwise.
    pred_labels: int Tensor
        Tensor of predicted labels, 2d if one-hot encoding is used,
        1d otherwise.
    one_hot: bool
        Whether one-hot encoding is used or not.

    Returns
    -------
    The accuracy as a float between 0 and 1.
    """

    if one_hot:
        eq = torch.eq(true_labels.max(1)[1], pred_labels.max(1)[1]).int()
    else:
        eq = torch.eq(true_labels, pred_labels).int()

    return eq.sum()/len(eq)

###############################################################################
#   utilities.py
###############################################################################


Dataset = collections.namedtuple('Dataset', ['data', 'labels'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


def loadDataSet(name=None, one_hot=True, train_size=1000, test_size=1000):
    """ Creates a 2D, 2 class dataset.

    If the name is not menitioned, the dataset is on [0, 1]**2 with label 1
    inside of the circle of radius 1/(2*np.pi)**0.5 centered on (0.5,0.5).
    If the name is "simple", the data set is a simple 2 point set with
    labels 0 and 1.

    Parameters
    ----------
    name : str
        Dataset to load, see above.
    one_hot : bool
        Whether to use one_hot label encoding.
    train_size : int
        Number of samples in training set.
    test_size : int
        Number of samples in test set.

    Returns
    -------
    A named tuple of type 'Datasets' containing the dataset.
    """
    if name is None:
        # Dataset generation
        test_data = torch.rand(test_size, 2)
        train_data = torch.rand(train_size, 2)
        test_labels = torch.norm(test_data - 0.5, 2, 1) < 1/(2*math.pi)**0.5
        train_labels = torch.norm(train_data - 0.5, 2, 1) < 1/(2*math.pi)**0.5
        # Boolean to int conversion
        test_labels = test_labels.float()
        train_labels = train_labels.float()
    elif name == "simple":
        test_data = FloatTensor([[1, 1], [-1, -1]])
        train_data = FloatTensor([[0.8, 0.9], [-0.8, -0.7]])
        test_labels = FloatTensor([0, 1])
        train_labels = FloatTensor([0, 1])

    if one_hot:
        # Convert labels to one-hot encoding
        test_labels = to_onehot(test_labels, 2)
        train_labels = to_onehot(train_labels, 2)

    # Combining datasets
    train = Dataset(data=train_data, labels=train_labels)
    test = Dataset(data=test_data, labels=test_labels)
    validation = Dataset(data=[], labels=[])

    return Datasets(train=train, test=test, validation=validation)


###############################################################################
#   Test code
###############################################################################


if __name__ == "__main__":
    data = loadDataSet(one_hot=True)

    # Create simple linear calssifier model
    lin1 = Linear(2, 25, std=0.1)
    relu1 = ReLU()
    lin2 = Linear(25, 25, std=0.1)
    relu2 = ReLU()
    lin3 = Linear(25, 25, std=0.1)
    relu3 = ReLU()
    lin4 = Linear(25, 2, std=0.1)
    sig1 = Sigmoid()
    model = Sequential([lin1, relu1, lin2, relu2, lin3, relu3, lin4, sig1])

    loss = MSE()
    optimizer = Adam()

    # Training model
    model.train(x=data.train.data,
                y=data.train.labels,
                batch_size=8,
                optimizer=optimizer,
                loss=loss,
                epochs=100)

    # Testing model on test data
    test_pred_labels = model.predict(x=data.test.data).round().int()
    train_pred_labels = model.predict(x=data.train.data).round().int()

    # Printing test accuracy
    test_acc = getAccuracy(data.test.labels.int(),
                           test_pred_labels,
                           one_hot=True)
    train_acc = getAccuracy(data.train.labels.int(),
                            train_pred_labels,
                            one_hot=True)
    print("Train Accuracy = {0:.2f}%, Test Accuracy = {1:.2f}%".
          format(train_acc*100, test_acc*100))
