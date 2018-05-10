from torch import sum


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
        return sum((output - labels)**2, dim=1)/labels.shape[1]

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
        return 2*(output - labels)/labels.shape[1]
