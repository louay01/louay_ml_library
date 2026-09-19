import numpy as np
from louay_ml_library.base import ActivationFunction

class ReLU(ActivationFunction):
    def activation(self, x):
        return np.maximum(0, x)

    def gradient(self, x):
        return np.where(x > 0, 1, 0)