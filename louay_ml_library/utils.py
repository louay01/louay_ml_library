import numpy as np

def validate_features(X, n_features=None):
    """Return a finite, nonempty (samples, features) floating-point array."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or 0 in X.shape or not np.all(np.isfinite(X)):
        raise ValueError("X must be a nonempty, finite 2D array.")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError("X has a different number of features than the training data.")
    return X


def validate_targets(y, n_samples):
    """Accept scalar targets as a vector or a single-column array."""
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1 or len(y) != n_samples:
        raise ValueError("y must contain one target per sample.")
    return y


def create_batches(x_train, y_train, batch_size, shuffle=True):
    if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if len(x_train) != len(y_train):
        raise ValueError("X and y must contain the same number of samples.")
    if shuffle:
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]

    batches = []
    for i in range(0, len(x_train), batch_size):
        batches.append((x_train[i:i+batch_size], y_train[i:i+batch_size]))
    return batches


# Xavier initialization with uniform distribution
def xavier_init_uniform(input_size, output_size):
        limit = np.sqrt(6 / (input_size + output_size))
        return np.random.uniform(-limit, limit, (output_size, input_size))

# Xavier initialization with normal distribution
def xavier_init_normal(input_size, output_size):
    stddev = np.sqrt(2 / (input_size + output_size))
    return np.random.normal(0, stddev, (output_size, input_size))

# He initialization with uniform distribution
def he_init_uniform(input_size, output_size):
    limit = np.sqrt(6 / input_size)
    return np.random.uniform(-limit, limit, (output_size, input_size))

# He initialization with normal distribution
def he_init_normal(input_size, output_size):
    stddev = np.sqrt(2 / input_size)
    return np.random.normal(0, stddev, (output_size, input_size))
