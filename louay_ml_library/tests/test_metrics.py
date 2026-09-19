import unittest
import numpy as np
from louay_ml_library.base import Model


class FixedClassifier(Model):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.array([1, 1, -1, -1])


class TestMetrics(unittest.TestCase):
    def test_negative_labels_and_column_targets(self):
        model = FixedClassifier()
        targets = np.array([1, -1, 1, -1])[:, None]
        for metric in ('accuracy', 'precision', 'recall', 'f1_score'):
            self.assertAlmostEqual(model.evaluate(None, targets, metric), .5)

    def test_rejects_broadcasting(self):
        with self.assertRaises(ValueError):
            FixedClassifier().evaluate(None, np.ones((4, 2)))
