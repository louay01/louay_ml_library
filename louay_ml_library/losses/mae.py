import numpy as np
from louay_ml_library.base import LossFunction

class MAE(LossFunction):
    def __init__(self):
        pass

    def compute_loss(self, y_true, y_pred):
        """
        Mean Absolute Error Loss.
        Parameters:
            y_true: Ground truth values (shape: [n_samples,]).
            y_pred: Predicted probabilities (shape: [n_samples,]).
        Returns:
            Scalar loss value (mean over all samples).
        """
        return np.mean(np.abs(y_pred - y_true))

    def gradient(self, y_true, y_pred):
        """
        Gradient of MAE Loss with respect to y_pred.
        Parameters:
            y_true: Ground truth values (shape: [n_samples,]).
            y_pred: Predicted probabilities (shape: [n_samples,]).
        Returns:
            Gradient with the same shape as y_pred.
        """
        return np.sign(y_pred - y_true) / y_true.size
