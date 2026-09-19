import contextlib
import io
import unittest
import numpy as np


class ModelTestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def fit_quietly(self, model, X, y):
        with contextlib.redirect_stdout(io.StringIO()):
            return model.fit(X, y)


def numerical_gradient(function, values, epsilon=1e-6):
    result = np.zeros_like(values, dtype=float)
    for index in np.ndindex(values.shape):
        plus, minus = values.copy(), values.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        result[index] = (function(plus) - function(minus)) / (2 * epsilon)
    return result
