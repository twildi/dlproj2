import matplotlib.pyplot as plt
import numpy as np


def plot2Dset(data, true_labels, pred_labels, w=None, b=None,
              xlim=[0, 1], ylim=[0, 1]):
    colors = ['r', 'b']

    # True label is the sourounding color
    plt.scatter(data[:, 0], data[:, 1],
                c=list(map(lambda x: colors[x], true_labels)))

    # Predicted label is color of central dot
    plt.scatter(data[:, 0], data[:, 1], linewidths=0.1, marker='.',
                c=list(map(lambda x: colors[x], pred_labels)))

    plt.axis('square')
    plt.xlim(xlim)
    plt.ylim(ylim)


def plotLinearSeperator(w, b):
    # PLot seperation line in interval x = [-1 1]
    plt.plot([0, 1], [-b/w[1], -(b + w[0])/w[1]])


def to_onehot(labels, nb_classes):
    nb_samples = len(labels)
    labels_onehot = np.zeros((nb_samples, nb_classes))
    labels_onehot[np.arange(nb_samples), labels] = 1
    return labels_onehot
