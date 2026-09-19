import numpy as np
from louay_ml_library import LinearRegression
from louay_ml_library.tests.test_helpers import ModelTestCase


class TestLinearRegression(ModelTestCase):
    def test_linear_fit_and_prediction_shape(self):
        X = np.linspace(-2, 2, 30).reshape(-1, 1)
        for optimizer in ('gradient descent', 'sgd'):
            model = LinearRegression(optimizer=optimizer)
            self.assertIs(self.fit_quietly(model, X, 3 * X[:, 0] + 2), model)
            prediction = model.predict(X)
            self.assertEqual(prediction.shape, (30,))
            np.testing.assert_allclose(prediction, 3 * X[:, 0] + 2, atol=1e-5)

    def test_polynomial_fit(self):
        X = np.linspace(-2, 2, 30).reshape(-1, 1)
        y = 2 * X[:, 0] ** 2 - X[:, 0] + 1
        model = LinearRegression(degree=2)
        self.fit_quietly(model, X, y)
        np.testing.assert_allclose(model.predict(X), y, atol=1e-5)

    def test_constant_feature_and_refit(self):
        model = LinearRegression()
        self.fit_quietly(model, np.ones((8, 1)), np.full(8, 3.))
        np.testing.assert_allclose(model.predict(np.ones((3, 1))), 3, atol=1e-5)
        X = np.arange(8.).reshape(-1, 1) + 100
        self.fit_quietly(model, X, X[:, 0])
        np.testing.assert_allclose(model.mean, X.mean(axis=0))
        np.testing.assert_allclose(model.predict(X), X[:, 0], atol=1e-5)

    def test_predict_before_fit(self):
        with self.assertRaises(ValueError):
            LinearRegression().predict([[1.]])
