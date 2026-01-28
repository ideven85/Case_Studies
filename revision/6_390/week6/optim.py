import abc

import numpy as np


class Optimizer(abc.ABC):
    def __init__(self, shape, **kwargs):
        """
        shape: tuple indicating the expected shape of the gradient
        """
        pass

    @abc.abstractmethod
    def get_grad_step(self, t, gradient):
        pass


class GD(Optimizer):
    def __init__(self, shape):
        pass

    def get_grad_step(self, t, gradient):
        return gradient


class Adam(Optimizer):
    def __init__(self, shape):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.B1 = 0.9
        self.B2 = 0.999
        self.eps = 1e-8

    def get_grad_step(self, t, gradient):
        self.m = self.B1 * self.m + (1 - self.B1) * gradient
        self.v = self.B2 * self.v + (1 - self.B2) * (gradient**2)
        m_hat = self.m / (1 - self.B1**t)
        v_hat = self.v / (1 - self.B2**t)
        return m_hat / np.sqrt(v_hat + self.eps)
