from abc import ABC, abstractmethod

class LossFunction(ABC):
    """Base class for loss functions."""
    def __init__(self):
        pass

    @abstractmethod
    def compute_loss(self, y_true, y_pred):
        pass

    @abstractmethod
    def gradient(self, y_true, y_pred):
        pass