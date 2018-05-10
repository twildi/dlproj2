from torch import sqrt


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
