# Louay ML Library

Machine-learning algorithms implemented from scratch with NumPy during my
master's degree, as part of a Machine Learning course supervised by
Prof. Hedi Tebia. The project focuses on understanding the algorithms and their
training procedures. It is an educational library with a small automated test suite.

## Implemented algorithms

- Linear and polynomial regression, trained with gradient descent or SGD.
- Binary perceptron.
- Decision-tree classification with entropy or Gini impurity.
- Binary SVM using simplified SMO, with linear and RBF kernels.
- Dense neural networks with configurable layers, activations, and losses.
- SGD with optional momentum, RMSProp, and Adam optimizers.
- Xavier and He initialization helpers.

## Installation

Requires Python 3.10 or newer. Install the published package:

```sh
python -m pip install louay-ml-library
```

For local development, from the project directory:

```sh
python -m venv .venv
```

Activate it on Windows PowerShell with `.venv\Scripts\Activate.ps1`, or on
macOS/Linux with `source .venv/bin/activate`, then install:

```sh
python -m pip install -e .
```

NumPy is the core dependency. The optional `roc_auc` evaluation metric uses
scikit-learn; install it with `python -m pip install -e ".[metrics]"`.

## Quick start

```python
import numpy as np
from louay_ml_library import LinearRegression

np.random.seed(42)
X = np.array([[0.], [1.], [2.], [3.]])
y = np.array([1., 3., 5., 7.])

model = LinearRegression(epochs=1000)
model.fit(X, y)
print(model.predict([[4.], [5.]]))  # Approximately [9., 11.]
```

Run `python examples/quickstart.py` after installation for regression and
multiclass neural-network examples.

## Neural networks

```python
from louay_ml_library import Network, DenseLayer, ActivationLayer
from louay_ml_library.activations import Tanh, Softmax
from louay_ml_library.losses import CrossEntropyLoss
from louay_ml_library.optimizers import Adam

network = Network(Adam(learning_rate=0.03), batch_size=4, epochs=150)
network.add(DenseLayer(2, 5))
network.add(ActivationLayer(Tanh()))
network.add(DenseLayer(5, 3))
network.add(ActivationLayer(Softmax()))
network.use(CrossEntropyLoss())

# X: (n_samples, 2); y_one_hot: (n_samples, 3)
# network.fit(X, y_one_hot)
# probabilities = network.predict(X)
# labels = probabilities.argmax(axis=1)
```

## API conventions

- Features have shape `(n_samples, n_features)`, including a single sample.
- Every model's `fit(X, y)` returns the model.
- Regression, tree, perceptron, and SVM predictions have shape `(n_samples,)`.
- Perceptron and SVM labels are `-1` and `1`. Trees also support string labels.
- Neural-network targets have shape `(n_samples, n_outputs)`; a vector is
  accepted for one output. Multiclass cross-entropy requires one-hot targets.
- Neural-network predictions are raw outputs of shape `(n_samples, n_outputs)`.
  Use sigmoid plus BCE for binary probabilities, or softmax plus cross-entropy
  for multiclass probabilities. Convert probabilities to labels explicitly.
- `loss_history_` stores the network's mean training loss for each epoch.
- `evaluate` is intended for predicted class labels. Precision, recall, and F1
  treat `1` as the positive class. SVM ROC AUC uses decision margins; models
  without a `decision_function` fall back to predicted labels.
- Random initialization and shuffling use NumPy's global random generator.
  Set `np.random.seed(...)` before training for repeatable examples.

Version 0.0.4 changes the old neural-network column-vector convention to rows
of samples. Dense weights remain `(n_outputs, n_inputs)` internally. It also
changes linear-regression predictions from a row matrix to a vector and makes
`Perceptron.predict(X)` handle the bias automatically.

## Tests

From the project directory after installation:

```sh
python -m unittest discover -s louay_ml_library/tests -v
```

Tests require NumPy and the standard library. They cover small learning tasks,
batch coverage, constant features, refitting, and numerical gradient checks.

## Scope and limitations

This project prioritizes readable implementations over performance. The SVM
uses Python loops and is intended for small datasets. Polynomial regression
expands each feature's powers without interaction terms. The library does not
aim for full scikit-learn API compatibility. Small synthetic tests are not a
benchmark of generalization performance.

## License

MIT. See [LICENCE.txt](LICENCE.txt).
