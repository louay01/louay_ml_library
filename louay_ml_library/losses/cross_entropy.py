import numpy as np
from louay_ml_library.base import LossFunction

class CrossEntropyLoss(LossFunction):
    def __init__(self):
        pass

    def compute_loss(self, y_true, y_pred):
        """
        Cross-Entropy Loss.
        Parameters:
            y_true: Ground truth labels (one-hot encoded) with shape (n_samples, n_classes).
            y_pred: Predicted probabilities with shape (n_samples, n_classes).
        Returns:
            Scalar loss value (mean over all samples).
        """
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def gradient(self, y_true, y_pred):
        """
        Gradient of Cross-Entropy Loss with respect to y_pred.
        Parameters:
            y_true: Ground truth labels (one-hot encoded) with shape (n_samples, n_classes).
            y_pred: Predicted probabilities with shape (n_samples, n_classes).
        Returns:
            Gradient with shape (n_samples, n_classes).
        """
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -y_true / y_pred / y_true.shape[0]
