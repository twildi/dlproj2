from framework import FloatTensor as Tensor
from utilites import to_onehot
import collections
import torch
import math

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
        test_data = Tensor([[1, 1], [-1, -1]])
        train_data = Tensor([[0.8, 0.9], [-0.8, -0.7]])
        test_labels = Tensor([0, 1])
        train_labels = Tensor([0, 1])

    if one_hot:
        # Convert labels to one-hot encoding
        test_labels = to_onehot(test_labels, 2)
        train_labels = to_onehot(train_labels, 2)

    # Combining datasets
    train = Dataset(data=train_data, labels=train_labels)
    test = Dataset(data=test_data, labels=test_labels)
    validation = Dataset(data=[], labels=[])

    return Datasets(train=train, test=test, validation=validation)
