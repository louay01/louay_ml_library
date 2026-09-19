import numpy as np
from louay_ml_library import SVM
from louay_ml_library.tests.test_helpers import ModelTestCase


class TestSVM(ModelTestCase):
    def test_linear_kernel(self):
        X = np.array([[-2., -1.], [-1., -2.], [1., 2.], [2., 1.]])
        y = np.array([-1, -1, 1, 1])
        model = SVM(max_passes=10)
        self.assertIs(model.fit(X, y), model)
        np.testing.assert_array_equal(model.predict(X), y)
        self.assertAlmostEqual(float(model.alphas @ y), 0., places=8)

    def test_rbf_xor(self):
        X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        y = np.array([-1, 1, 1, -1])
        model = SVM(C=10, kernel='rbf', gamma=2, max_passes=20).fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)

    def test_refit_recomputes_gamma(self):
        X = np.array([[-2.], [-1.], [1.], [2.]])
        y = np.array([-1, -1, 1, 1])
        model = SVM(kernel='rbf', gamma='scale')
        model.fit(X, y)
        model.b = 100
        np.random.seed(7)
        model.fit(10 * X, y)
        fresh = SVM(kernel='rbf', gamma='scale')
        np.random.seed(7)
        fresh.fit(10 * X, y)
        self.assertEqual(model.gamma, 'scale')
        np.testing.assert_allclose(model.decision_function(X), fresh.decision_function(X))

    def test_rejects_single_sample_and_invalid_labels(self):
        for X, y in [([[1]], [1]), ([[0], [1]], [0, 1])]:
            with self.assertRaises(ValueError):
                SVM().fit(X, y)

    def test_zero_margin_returns_a_class(self):
        model = SVM().fit([[1], [1]], [-1, 1])
        self.assertIn(model.predict([[1]])[0], (-1, 1))
