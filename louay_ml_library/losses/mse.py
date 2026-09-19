import numpy as np
from louay_ml_library.base import LossFunction

class MSE(LossFunction):
    def __init__(self):
        pass

    def compute_loss(self, y_true, y_pred):
        """
        Mean Squared Error Loss.
        Parameters:
            y_true: Ground truth values (shape: [n_samples,]).
            y_pred: Predicted probabilities (shape: [n_samples,]).
        Returns:
            Scalar loss value (mean over all samples).
        """
        return np.mean((y_pred - y_true) ** 2)

    def gradient(self, y_true, y_pred):
        """
        Gradient of MSE Loss with respect to y_pred.
        Parameters:
            y_true: Ground truth values (shape: [n_samples,]).
            y_pred: Predicted probabilities (shape: [n_samples,]).
        Returns:
            Gradient with the same shape as y_pred.
        """
        return 2 * (y_pred - y_true) / y_true.size
