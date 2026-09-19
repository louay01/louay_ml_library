import numpy as np
from louay_ml_library import Perceptron
from louay_ml_library.tests.test_helpers import ModelTestCase


class TestPerceptron(ModelTestCase):
    def test_separable_data_with_bias(self):
        X = np.array([[1.], [2.], [4.], [5.]])
        y = np.array([-1, -1, 1, 1])
        model = Perceptron(epochs=100)
        self.assertIs(model.fit(X, y), model)
        np.testing.assert_array_equal(model.predict(X), y)
        self.assertEqual(model.evaluate(X, y[:, None]), 1.)

    def test_invalid_labels(self):
        with self.assertRaises(ValueError):
            Perceptron().fit([[0], [1]], [0, 1])

    def test_predict_before_fit(self):
        with self.assertRaises(ValueError):
            Perceptron().predict([[0]])
