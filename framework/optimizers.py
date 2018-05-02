class Optimizer(object):
    def apply(self):
        raise NotImplementedError


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
                p.value -= self.eta * p.gradient


class Adam(Optimizer):
    """
    Adam optimzer.
    """
    pass
