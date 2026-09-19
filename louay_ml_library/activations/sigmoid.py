import numpy as np
from louay_ml_library.base import ActivationFunction

class Sigmoid(ActivationFunction):
    def activation(self, x):
        return np.exp(-np.logaddexp(0, -np.asarray(x, dtype=float)))

    def gradient(self, x):
        return (self.activation(x)) * (1 - self.activation(x))
