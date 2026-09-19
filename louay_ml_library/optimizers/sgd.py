import numpy as np
from louay_ml_library.base import Optimizer

class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer with optional momentum.
    """
    def __init__(self, learning_rate=0.01, momentum=0.0):
        """
        Initializes the SGD optimizer.

        Parameters:
        - learning_rate (float): The step size for parameter updates.
        - momentum (float): Momentum factor (default is 0, meaning no momentum).
        """
        super().__init__(learning_rate)
        self.momentum = momentum

    def step(self, layers):
        """
        Performs an SGD update with momentum.

        Parameters:
        - layers (np.ndarray): All the layers of the network.
        """

        for layer in layers:
            layer.v_w = self.momentum * layer.v_w + (1 - self.momentum) * layer.grad_weights
            layer.v_b = self.momentum * layer.v_b + (1 - self.momentum) * layer.grad_bias

            layer.weights -= self.learning_rate * layer.v_w
            layer.bias -= self.learning_rate * layer.v_b