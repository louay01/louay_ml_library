import numpy as np
from louay_ml_library.base import ActivationFunction

class Sign(ActivationFunction):
    def activation(self, x):
        """
        Compute the Sign activation output: 1 for input >= 0, and -1 for input < 0.
        """
        return np.where(x >= 0, 1, -1)

    def gradient(self, x):
        """
        The derivative of the Sign activation is 0 everywhere except at 0.
        Since the derivative is undefined at 0, we will return 0 for all values.
        """
        return np.zeros_like(x) 