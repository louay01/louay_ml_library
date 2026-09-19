import numpy as np
from louay_ml_library.base import LossFunction 

class BCELoss(LossFunction):
    def __init__(self):
        pass

    def compute_loss(self, y_true, y_pred):
        """
        Binary Cross-Entropy Loss.
        Parameters:
            y_true: Ground truth values (shape: [n_samples,]).
            y_pred: Predicted probabilities (shape: [n_samples,]).
        Returns:
            BCE loss (scalar).
        """
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def gradient(self, y_true, y_pred):
        """
        Gradient of BCE Loss with respect to y_pred.
        Parameters:
            y_true: Ground truth values (shape: [n_samples,]).
            y_pred: Predicted probabilities (shape: [n_samples,]).
        Returns:
            Gradient (shape: [n_samples,]).
        """
        # Clip predictions to avoid division by zero
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.size
