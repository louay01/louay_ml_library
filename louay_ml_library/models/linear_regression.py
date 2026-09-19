import numpy as np
from louay_ml_library.base import Model
from louay_ml_library.utils import validate_features, validate_targets

class LinearRegression(Model):
    def __init__(self, degree=1, epochs=1000, learning_rate=0.01, optimizer='gradient descent', criterion='mse'):
        """
        Initialize Regression model.

        Parameters:
        - degree (int): Degree of polynomial regression. Set to 1 for linear regression.
        - epochs (int): Number of training iterations.
        - learning_rate (float): Learning rate for optimization.
        - optimizer (str): Optimization method, 'gradient descent' or 'sgd'.
        - criterion (str): Loss function, default is 'mse'.
        """
        self.degree = degree
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer.lower()
        if criterion != 'mse':
            raise ValueError("Currently, only 'mse' is supported for criterion.")
        self.w = None
        self.mean = None
        self.std = None

    def _make_polynomial_features(self, x):
        if self.degree == 1:  # Linear regression
            return x
        polynomial_features = x.copy()
        for i in range(2, self.degree + 1):
            polynomial_features = np.hstack((polynomial_features, x ** i))
        return polynomial_features

    def _normalize_features(self, x):
        if self.mean is None or self.std is None:
            self.mean = np.mean(x, axis=0)
            self.std = np.std(x, axis=0)
            self.std = np.where(self.std == 0, 1.0, self.std)
        return (x - self.mean) / self.std

    def _make_data_matrix(self, x):
        x = np.hstack((np.ones((x.shape[0], 1)), x))
        return x

    def fit(self, x, y):
        """
        Fit the model to the training data.
        Parameters:
            X (np.ndarray): The input features (2D array where each row is a sample).
            y (np.ndarray): The target labels (1D array with corresponding labels).
        """
        x = validate_features(x)
        y = validate_targets(y, len(x)).astype(float)
        if not np.all(np.isfinite(y)):
            raise ValueError("y must be finite.")
        self.n_features_in_ = x.shape[1]
        self.mean = self.std = None
        x = self._make_polynomial_features(x)
        x = self._normalize_features(x)
        x = self._make_data_matrix(x)
        y = y.reshape(1, -1)[0]
        self.w = np.random.rand(1, x.shape[1])

        for epoch in range(self.epochs):
            y_pred = self._predict(x)
            gradients = -(2 / len(y)) * np.dot((y - y_pred), x)

            if self.optimizer == 'gradient descent':
                self._gradient_descent_update(gradients)
            elif self.optimizer == 'sgd':
                self._stochastic_gradient_descent_update(x, y)
            else:
                raise ValueError("Invalid optimizer. Use 'gradient descent' or 'sgd'.")

            if epoch % 100 == 0:
                loss = self._mse_loss(y, y_pred)
                print(f'Epoch {epoch}, Loss: {loss}')
        return self

    def _gradient_descent_update(self, gradients):
        self.w -= self.learning_rate * gradients

    def _stochastic_gradient_descent_update(self, x, y):
        for i in range(len(y)):
            xi = x[i, :].reshape(1, -1)
            yi = y[i]
            yi_pred = np.dot(self.w, xi.T)
            gradient = -(2 * (yi - yi_pred)) * xi
            self.w -= self.learning_rate * gradient

    def _predict(self, x):
        return np.dot(self.w, x.T)

    def predict(self, x):
        if self.w is None:
            raise ValueError("Call fit before predict.")
        x = validate_features(x, self.n_features_in_)
        x = self._make_polynomial_features(x)
        x = self._normalize_features(x)
        x = self._make_data_matrix(x)
        return self._predict(x).ravel()

    def _mse_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
