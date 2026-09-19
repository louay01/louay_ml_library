import numpy as np
from louay_ml_library.base import Model
from louay_ml_library.utils import validate_features, validate_targets

class DecisionTreeClassifier(Model):
    def __init__(self, max_depth, criterion='entropy'):
        """
        Initializes the decision tree model with a specified max depth and criterion.
        Parameters:
            max_depth (int): The maximum depth of the tree.
            criterion (str): The criterion for splitting the nodes, either 'entropy' or 'gini'.
        """
        if not isinstance(max_depth, (int, np.integer)) or max_depth < 0:
            raise ValueError("max_depth must be a nonnegative integer.")
        if criterion not in ('entropy', 'gini'):
            raise ValueError("criterion must be 'entropy' or 'gini'.")
        self.max_depth = max_depth
        self.criterion = criterion
        self.tree = None

    def fit(self, X, y):
        """
        Fit the decision tree model to the training data.
        Parameters:
            X (np.ndarray): The input features (2D array where each row is a sample).
            y (np.ndarray): The target labels (1D array with corresponding labels).
        """
        X = validate_features(X)
        y = validate_targets(y, len(X))
        self.n_features_in_ = X.shape[1]
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth: int):
        labels, counts = np.unique(y, return_counts=True)
        n_labels = len(labels)
        leaf_value = labels[np.argmax(counts)]
        
        # If we've reached the maximum depth or there's only one label, create a leaf
        if depth >= self.max_depth or n_labels == 1:
            return ('leaf', leaf_value)  # Return a tuple ('leaf', class_value) for leaf nodes
        
        # Otherwise, find the best split and recursively build the tree
        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            return ('leaf', leaf_value)
        left_indices = X[:, best_feature] < best_threshold
        right_indices = ~left_indices
        
        # Recursively build left and right subtrees
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        # Return a decision node as a tuple (feature, threshold, left_subtree, right_subtree)
        return (best_feature, best_threshold, left_subtree, right_subtree)

    def _find_best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        n_features = X.shape[1]
        
        for i in range(n_features):
            thresholds = np.unique(X[:, i]) 
            for t in range(len(thresholds)-1):
                threshold = np.mean([thresholds[t], thresholds[t+1]])
                left_indices = X[:, i] < threshold
                right_indices = ~left_indices
                gain = self._information_gain(y, y[left_indices], y[right_indices])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = i
                    best_threshold = threshold
        
        return best_feature, best_threshold

    def _information_gain(self, y, left_y, right_y):
        if self.criterion == 'entropy':
            return self._entropy(y) - (len(left_y) / len(y)) * self._entropy(left_y) - (len(right_y) / len(y)) * self._entropy(right_y)
        elif self.criterion == 'gini':
            return self._gini_index(y) - (len(left_y) / len(y)) * self._gini_index(left_y) - (len(right_y) / len(y)) * self._gini_index(right_y)

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        prob = counts / len(y)
        return np.sum(-prob * np.log2(prob + 1e-9))  # Small value to avoid log(0)

    def _gini_index(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities**2)

    def predict(self, X):
        """
        Predicts the target labels for the input data.
        Parameters:
            X (np.ndarray): The input data to make predictions on.
        Returns:
            predictions (np.ndarray): The predicted labels.
        """
        if self.tree is None:
            raise ValueError("Call fit before predict.")
        X = validate_features(X, self.n_features_in_)
        predictions = [self._predict_one(sample, self.tree) for sample in X]
        return np.array(predictions)

    def _predict_one(self, x, tree):
        # Check if it's a leaf node (a tuple starting with 'leaf')
        if isinstance(tree, tuple) and tree[0] == 'leaf':
            return tree[1]  # Return the class value (leaf value)
        
        # Otherwise, it's a decision node, so recursively go to the left or right subtree
        feature, threshold, left_subtree, right_subtree = tree
        
        if x[feature] < threshold:
            return self._predict_one(x, left_subtree)  # Go to left subtree
        else:
            return self._predict_one(x, right_subtree)  # Go to right subtree
