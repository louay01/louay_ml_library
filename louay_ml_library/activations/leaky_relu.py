import numpy as np
from louay_ml_library.base import ActivationFunction

class LeakyReLU(ActivationFunction):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def activation(self, x):
        self.input = x
        return np.where(x > 0, x, self.alpha * x)

    def gradient(self, x):
        return np.where(x > 0, 1, self.alpha)