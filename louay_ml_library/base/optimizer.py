from abc import ABC, abstractmethod
import numpy as np

class Optimizer(ABC):
    """
    Abstract base class for all optimizers.
    """
    def __init__(self, learning_rate=0.01):
        """
        Initializes the optimizer with a learning rate.

        Parameters:
        - learning_rate (float): The step size for parameter updates.
        """
        self.learning_rate = learning_rate

    @abstractmethod
    def step(self, layers):
        """
        Updates the model parameters (weights) based on gradients.

        Parameters:
        - layers (np.ndarray): All the layers of the network.
        """
        pass
