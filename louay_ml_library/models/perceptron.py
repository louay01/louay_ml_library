import numpy as np
from louay_ml_library.activations import Sign
from louay_ml_library.base import Model
from louay_ml_library.utils import validate_features, validate_targets

class Perceptron(Model):
    def __init__(self, epochs=50, learning_rate=0.1, activation=None):
        self.epochs = epochs
        self.activation = Sign() if activation is None else activation
        self.learning_rate = learning_rate
        self.w = None
    
    def predict(self, x):
        if self.w is None:
            raise ValueError("Call fit before predict.")
        x = validate_features(x, self.w.shape[1] - 1)
        x = np.column_stack((np.ones(len(x)), x))
        return self.activation.activation(np.dot(self.w, x.T)).ravel()
    
    def fit(self, x, y):
        x = validate_features(x)
        y = validate_targets(y, len(x))
        if not np.all(np.isin(y, [-1, 1])):
            raise ValueError("Perceptron labels must be -1 or 1.")
        N, features_nb = x.shape
        self.w = np.random.rand(1, features_nb + 1)
        x = np.hstack((np.ones((N, 1)), x))
        y = y.reshape(1, -1)[0]
        for epoch in range(self.epochs):
            for i in range(N):
                y_pred = self.activation.activation(np.dot(self.w, x[i]))
                self.w += (self.learning_rate*(y[i] - y_pred)*x[i])
        return self
