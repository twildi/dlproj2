import collections
import numpy as np
from utilites import to_onehot

Dataset = collections.namedtuple('Dataset', ['data', 'labels'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


def loadDataSet(name=None, one_hot=True, train_size=1000, test_size=1000):
    """ Creates a 2D, 2 class dataset. 
    
    If the name is not menitioned, the 
    dataset is on [0, 1]**2 with label 1 inside of the circle of radius
    1/(2*np.pi)**0.5 centered on (0.5,0.5). If the name is "simple", the data
    set is a simple 2 point set with labels 0 and 1.
    
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
        test_data = np.random.rand(test_size, 2)
        train_data = np.random.rand(train_size, 2)
        test_labels = np.linalg.norm(test_data - 0.5, axis=1) < 1/(2*np.pi)**0.5
        train_labels = np.linalg.norm(train_data - 0.5, axis=1) < 1/(2*np.pi)**0.5
        # Boolean to int conversion
        test_labels = test_labels.astype('int')
        train_labels = train_labels.astype('int')
    elif name == "simple":
        test_data = np.array([[1, 1], [-1, -1]])
        train_data = np.array([[0.8, 0.9], [-0.8, -0.7]])
        test_labels = np.array([0, 1])
        train_labels = np.array([0, 1])

    if one_hot:
        # Convert labels to one-hot encoding
        test_labels = to_onehot(test_labels, 2)
        train_labels = to_onehot(train_labels, 2)

    # Combining datasets
    train = Dataset(data=train_data, labels=train_labels)
    test = Dataset(data=test_data, labels=test_labels)
    validation = Dataset(data=[], labels=[])

    return Datasets(train=train, test=test, validation=validation)
