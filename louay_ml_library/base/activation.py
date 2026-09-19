from abc import ABC, abstractmethod

class ActivationFunction(ABC):
    def backward(self, x, output_gradient):
        """Propagate gradients through an elementwise activation."""
        return self.gradient(x) * output_gradient

    @abstractmethod
    def activation(self, x):
        """
        Compute the activation output for the input.
        """
        pass

    @abstractmethod
    def gradient(self, x):
        """
        Compute the derivative of the activation function.
        """
        pass
