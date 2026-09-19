import numpy as np
from louay_ml_library.base import Model
from louay_ml_library.utils import xavier_init_normal
from louay_ml_library.utils import create_batches, validate_features

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError
    
    def backward(self, output_gradient):
        raise NotImplementedError
    
class DenseLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = xavier_init_normal(input_size, output_size)
        self.bias = np.zeros((output_size, 1))
        self.grad_weights = None
        self.grad_bias = None

        self.v_w = np.zeros_like(self.weights)
        self.v_b = np.zeros_like(self.bias)

        self.s_w = np.zeros_like(self.weights)
        self.s_b = np.zeros_like(self.bias)

    def forward(self, input):
        self.input = input
        return np.dot(self.input, self.weights.T) + self.bias.T
    
    def backward(self, output_gradient):
        self.grad_weights = np.dot(output_gradient.T, self.input)
        self.grad_bias = np.sum(output_gradient, axis=0).reshape(-1, 1)
        return np.dot(output_gradient, self.weights)
    
class ActivationLayer(Layer):
    def __init__(self, activation_fn):
        self.activation_fn = activation_fn

    def forward(self, input):
        self.input = input
        return self.activation_fn.activation(self.input)
    
    def backward(self, output_gradient):
        return self.activation_fn.backward(self.input, output_gradient)

class Network(Model):
    def __init__(self, optimizer, batch_size=1, epochs=1000):
        if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if not isinstance(epochs, (int, np.integer)) or epochs <= 0:
            raise ValueError("epochs must be a positive integer.")
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.layers = []
        self.loss_fn = None
        self.loss_history_ = []
        self._fitted = False

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss_fn):
        self.loss_fn = loss_fn

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def predict(self, X):
        """Return raw outputs with shape (n_samples, n_outputs)."""
        if not self._fitted:
            raise ValueError("Call fit before predict.")
        return self.forward(validate_features(X, self.n_features_in_))
    
    def fit(self, X, Y):
        """Train on row-major batches; Y is a vector or a target matrix."""
        X = validate_features(X)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        if Y.ndim != 2 or Y.shape[0] != len(X) or Y.shape[1] == 0 or not np.all(np.isfinite(Y)):
            raise ValueError("Y must be a finite target matrix with one row per sample.")
        if not self.layers or self.loss_fn is None:
            raise ValueError("Add layers and select a loss with use() before fit.")
        self.n_features_in_ = X.shape[1]
        if self.forward(X[:1]).shape != Y[:1].shape:
            raise ValueError("Network output shape must match the target shape.")
        self.loss_history_ = []
        for epoch in range(self.epochs):
            error = 0.0
            for x_batch, y_batch in create_batches(X, Y, self.batch_size):
                output = self.forward(x_batch)
                error += self.loss_fn.compute_loss(y_batch, output) * len(x_batch)
                self.backward(self.loss_fn.gradient(y_batch, output))
                self.optimizer.step(self._parameters())
            self.loss_history_.append(error / len(X))
        self._fitted = True
        return self
    
    def _parameters(self):
        return [layer for layer in self.layers if isinstance(layer, DenseLayer)]
