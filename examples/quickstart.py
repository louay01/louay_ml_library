"""Run after installing the project: python examples/quickstart.py."""
import numpy as np

from louay_ml_library import LinearRegression, Network, DenseLayer, ActivationLayer
from louay_ml_library.activations import Tanh, Softmax
from louay_ml_library.losses import CrossEntropyLoss
from louay_ml_library.optimizers import Adam


def main():
    np.random.seed(42)
    X = np.arange(4.).reshape(-1, 1)
    regression = LinearRegression().fit(X, 2 * X[:, 0] + 1)
    print("Regression predictions (expected 9 and 11):", regression.predict([[4.], [5.]]))

    X = np.array([[-2., 0.], [-1., 0.], [2., 0.], [1., 0.], [0., 2.], [0., 1.]])
    labels = np.array([0, 0, 1, 1, 2, 2])
    network = Network(Adam(.03), batch_size=4, epochs=150)
    network.add(DenseLayer(2, 5))
    network.add(ActivationLayer(Tanh()))
    network.add(DenseLayer(5, 3))
    network.add(ActivationLayer(Softmax()))
    network.use(CrossEntropyLoss())
    network.fit(X, np.eye(3)[labels])
    predictions = network.predict(X).argmax(axis=1)
    print("Neural-network training accuracy:", np.mean(predictions == labels))


if __name__ == '__main__':
    main()
