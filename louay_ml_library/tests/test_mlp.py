import numpy as np
from louay_ml_library import Network, DenseLayer, ActivationLayer
from louay_ml_library.activations import Softmax, Tanh, Sigmoid
from louay_ml_library.losses import MSE, CrossEntropyLoss, BCELoss, MAE
from louay_ml_library.optimizers import SGD, Adam, RMSProp
from louay_ml_library.tests.test_helpers import ModelTestCase, numerical_gradient


class TestMLP(ModelTestCase):
    def test_every_sample_is_used_each_epoch(self):
        class RecordingLoss(MSE):
            def __init__(self):
                self.seen = []
            def compute_loss(self, y, prediction):
                self.seen.extend(y[:, 0].tolist())
                return super().compute_loss(y, prediction)
        loss = RecordingLoss()
        model = Network(SGD(0), batch_size=4, epochs=3)
        model.add(DenseLayer(1, 1))
        model.use(loss)
        X = np.arange(10.).reshape(-1, 1)
        self.assertIs(model.fit(X, X), model)
        for epoch in range(3):
            self.assertEqual(sorted(loss.seen[epoch * 10:(epoch + 1) * 10]), list(range(10)))
        self.assertEqual(len(model.loss_history_), 3)

    def test_dense_gradients(self):
        layer = DenseLayer(2, 3)
        X = np.array([[.2, -.4], [.7, .1]])
        upstream = np.array([[.1, .2, -.3], [.4, -.5, .6]])
        layer.forward(X)
        dx = layer.backward(upstream)
        expected_x = numerical_gradient(lambda value: np.sum(layer.forward(value) * upstream), X)
        np.testing.assert_allclose(dx, expected_x, atol=1e-8)
        for name, actual in [('weights', layer.grad_weights.copy()), ('bias', layer.grad_bias.copy())]:
            original = getattr(layer, name).copy()
            def objective(value):
                setattr(layer, name, value)
                return np.sum(layer.forward(X) * upstream)
            expected = numerical_gradient(objective, original)
            setattr(layer, name, original)
            np.testing.assert_allclose(actual, expected, atol=1e-8)

    def test_softmax_cross_entropy_chain_gradient(self):
        logits = np.array([[.2, -.4, .5], [.1, .8, -.2]])
        targets = np.array([[1., 0., 0.], [0., 1., 0.]])
        activation = ActivationLayer(Softmax())
        loss = CrossEntropyLoss()
        prediction = activation.forward(logits)
        actual = activation.backward(loss.gradient(targets, prediction))
        expected = numerical_gradient(lambda value: loss.compute_loss(targets, Softmax().activation(value)), logits)
        np.testing.assert_allclose(actual, expected, atol=1e-8)
        jacobian = Softmax().gradient(logits)
        np.testing.assert_allclose(np.einsum('bij,bj->bi', jacobian, loss.gradient(targets, prediction)), actual)

    def test_loss_gradients(self):
        targets = np.array([[1., 0.], [0., 1.]])
        prediction = np.array([[.7, .3], [.4, .6]])
        for loss in (MSE(), BCELoss(), CrossEntropyLoss(), MAE()):
            expected = numerical_gradient(lambda value: loss.compute_loss(targets, value), prediction)
            np.testing.assert_allclose(loss.gradient(targets, prediction), expected, atol=1e-8)
        np.testing.assert_array_equal(MAE().gradient(targets, targets), np.zeros_like(targets))

    def test_regression_with_all_optimizers(self):
        X = np.linspace(-1, 1, 20).reshape(-1, 1)
        y = 2 * X + 1
        for optimizer in (SGD(.05), Adam(.05), RMSProp(.01)):
            model = Network(optimizer, batch_size=6, epochs=250)
            model.add(DenseLayer(1, 1))
            model.use(MSE())
            model.fit(X, y)
            self.assertEqual(model.predict(X).shape, y.shape)
            self.assertLess(np.mean((model.predict(X) - y) ** 2), .01)

    def test_multiclass_learning(self):
        X = np.array([[-2., 0.], [-1., 0.], [2., 0.], [1., 0.], [0., 2.], [0., 1.]])
        labels = np.array([0, 0, 1, 1, 2, 2])
        model = Network(Adam(.03), batch_size=4, epochs=150)
        model.add(DenseLayer(2, 5))
        model.add(ActivationLayer(Tanh()))
        model.add(DenseLayer(5, 3))
        model.add(ActivationLayer(Softmax()))
        model.use(CrossEntropyLoss())
        model.fit(X, np.eye(3)[labels])
        probabilities = model.predict(X)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.)
        np.testing.assert_array_equal(probabilities.argmax(axis=1), labels)

    def test_sigmoid_extreme_inputs(self):
        with np.errstate(over='raise', invalid='raise'):
            np.testing.assert_allclose(Sigmoid().activation(np.array([-1000., 0., 1000.])), [0., .5, 1.])

    def test_rejects_invalid_shapes(self):
        model = Network(SGD(), epochs=1)
        model.add(DenseLayer(2, 2))
        model.use(MSE())
        with self.assertRaises(ValueError):
            model.fit(np.ones((3, 2)), np.ones(3))
        with self.assertRaises(ValueError):
            Network(SGD(), batch_size=0)
