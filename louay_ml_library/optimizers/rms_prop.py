import numpy as np
from louay_ml_library.base import Optimizer

class RMSProp(Optimizer):
    """
    RMSProp optimizer.
    """
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        """
        Initializes the RMSProp optimizer.

        Parameters:
        - learning_rate (float): The step size for parameter updates.
        - beta (float): Decay rate for the moving average of squared gradients (default is 0.9).
        - epsilon (float): Small value to prevent division by zero (default is 1e-8).
        """
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.s = None

    def step(self, layers):
        """
        Performs an RMSProp update.

        Parameters:
        - layers (np.ndarray): All the layers of the network.
        """

        for layer in layers:
          layer.s_w = self.beta * layer.s_w + (1 - self.beta) * layer.grad_weights ** 2
          layer.weights -= self.learning_rate * layer.grad_weights / (np.sqrt(layer.s_w) + self.epsilon)
          
          layer.s_b = self.beta * layer.s_b + (1 - self.beta) * layer.grad_bias ** 2
          layer.bias -= self.learning_rate * layer.grad_bias / (np.sqrt(layer.s_b) + self.epsilon)
