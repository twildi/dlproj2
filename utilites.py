#import matplotlib.pyplot as plt
import torch


#def plot2Dset(data, true_labels, pred_labels, w=None, b=None,
#              xlim=[0, 1], ylim=[0, 1]):
#    colors = ['r', 'b']
#
#    # True label is the sourounding color
#    plt.scatter(data[:, 0], data[:, 1],
#                c=list(map(lambda x: colors[int(x)], true_labels)))
#
#    # Predicted label is color of central dot
#    plt.scatter(data[:, 0], data[:, 1], linewidths=0.1, marker='.',
#                c=list(map(lambda x: colors[int(x)], pred_labels)))
#
#    plt.axis('square')
#    plt.xlim(xlim)
#    plt.ylim(ylim)
#
#
#def plotLinearSeperator(w, b):
#    # PLot seperation line in interval x = [-1 1]
#    plt.plot([0, 1], [-b/w[1], -(b + w[0])/w[1]])


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
    labels_onehot = labels.view(-1, 1) == torch.arange(nb_classes).view(1, -1)
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
