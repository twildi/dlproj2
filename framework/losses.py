from torch import sum

class Loss(object):
    pass


class MSE(Loss):
    def loss(self, output, labels):
        return sum((output - labels)**2, dim=1)/labels.shape[1]

    def dloss(self, output, labels):
        """
        Derivative of the loss as function of ouput of the network.
        """
        return 2*(output - labels)/labels.shape[1]
