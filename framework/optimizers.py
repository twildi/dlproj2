import numpy as np

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
    def __init__(self,eta=1e-3,rho1=0.9,rho2=0.999,delta=1e-8):
        self.eta = eta
        self.rho1 = rho1
        self.rho2 = rho2
        self.delta = delta
        self.t = 0
        
    def apply(self,layers):
        self.t = self.t + 1
        for layer in layers:
            param = layer.get_param()
            for p in param:
                new_g = p.gradient
                new_m = self.rho1*p.m + (1-self.rho1)*new_g
                new_v = self.rho2*p.v + (1-self.rho2)*new_g**2
                p.g = new_g
                p.m = new_m
                p.v = new_v
                
                m_hat = p.m/(1-self.rho1**self.t)
                v_hat = p.v/(1-self.rho2**self.t)
                
                p.value -= self.eta*m_hat/(np.sqrt(v_hat)+self.delta)
                