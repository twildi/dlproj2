import collections
import numpy as np
from utilites import to_onehot

Dataset = collections.namedtuple('Dataset', ['data', 'labels'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


def loadDataSet(name=None, one_hot=True):
    if name is None:
        # Dataset generation
        test_data = np.random.rand(1000, 2)
        train_data = np.random.rand(1000, 2)
        test_labels = np.linalg.norm(test_data, axis=1) < 0.8  # 1/(2*np.pi)**0.5
        train_labels = np.linalg.norm(train_data, axis=1) < 0.8  # 1/(2*np.pi)**0.5
        # Boolean to int conversion
        test_labels = test_labels.astype('int')
        train_labels = train_labels.astype('int')
    elif name == "simple":
        test_data = np.array([[1, 1], [-1, -1]])
        train_data = np.array([[1.1, 1.2], [-1.1, -0.9]])
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
