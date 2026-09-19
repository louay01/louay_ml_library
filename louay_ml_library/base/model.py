from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def evaluate(self, X_val, y_val, metric="accuracy"):
        """
        Evaluates the performance of the model on the validation data using a specified metric.

        Paramters:
            X_val (np.ndarray): The validation input data (2D array where each row is a sample).
            y_val (np.ndarray): The validation target labels (1D array with corresponding labels).
            metric (str): The evaluation metric to use. Available metrics:
                - "accuracy": Percentage of correctly classified samples.
                - "precision": Ratio of true positives to the sum of true positives and false positives.
                - "recall": Ratio of true positives to the sum of true positives and false negatives.
                - "f1_score": The harmonic mean of precision and recall (2 * (precision * recall) / (precision + recall)).
                - "roc_auc": Area under the Receiver Operating Characteristic curve (for binary classification).
        
        Returns:
            score (float): The evaluation score based on the specified metric.
        
        Raises:
            ValueError: If the specified metric is not recognized.
        """
        y_pred = np.asarray(self.predict(X_val))
        y_val = np.asarray(y_val)
        if y_val.ndim == 2 and y_val.shape[1] == 1:
            y_val = y_val.ravel()
        if y_pred.ndim != 1 or y_val.shape != y_pred.shape:
            raise ValueError("evaluate requires one predicted label per sample; convert network outputs explicitly.")
        
        if metric == "accuracy":
            return np.mean(y_pred == y_val)
        
        elif metric == "precision":
            tp = np.sum((y_pred == 1) & (y_val == 1))
            fp = np.sum((y_pred == 1) & (y_val != 1))
            return tp / (tp + fp) if tp + fp > 0 else 0
        
        elif metric == "recall":
            tp = np.sum((y_pred == 1) & (y_val == 1))
            fn = np.sum((y_pred != 1) & (y_val == 1))
            return tp / (tp + fn) if tp + fn > 0 else 0
        
        elif metric == "f1_score":
            precision = self.evaluate(X_val, y_val, metric="precision")
            recall = self.evaluate(X_val, y_val, metric="recall")
            return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        
        elif metric == "roc_auc":
            from sklearn.metrics import roc_auc_score
            scores = self.decision_function(X_val) if hasattr(self, 'decision_function') else y_pred
            return roc_auc_score(y_val, scores)
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
