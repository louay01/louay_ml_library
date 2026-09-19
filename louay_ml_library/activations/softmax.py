import numpy as np
from louay_ml_library.base import ActivationFunction

class Softmax(ActivationFunction):
    def activation(self, x):
        exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def gradient(self, x):
        """Return the full Jacobian for each sample, including cross terms."""
        s = self.activation(x)
        return s[..., :, None] * np.eye(s.shape[-1]) - s[..., :, None] * s[..., None, :]

    def backward(self, x, output_gradient):
        """Multiply by the Jacobian without allocating it."""
        s = self.activation(x)
        return s * (output_gradient - np.sum(output_gradient * s, axis=-1, keepdims=True))
