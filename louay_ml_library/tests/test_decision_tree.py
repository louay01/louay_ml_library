import numpy as np
from louay_ml_library import DecisionTreeClassifier
from louay_ml_library.tests.test_helpers import ModelTestCase


class TestDecisionTree(ModelTestCase):
    def test_xor_with_both_criteria(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])
        for criterion in ('entropy', 'gini'):
            model = DecisionTreeClassifier(2, criterion)
            self.assertIs(model.fit(X, y), model)
            np.testing.assert_array_equal(model.predict(X), y)

    def test_unsplittable_node_returns_majority(self):
        model = DecisionTreeClassifier(3).fit(np.ones((5, 2)), [-1, 1, 1, -1, 1])
        np.testing.assert_array_equal(model.predict([[1, 1]]), [1])

    def test_string_labels_and_zero_depth(self):
        model = DecisionTreeClassifier(0).fit([[0], [1], [2]], ['cat', 'dog', 'cat'])
        np.testing.assert_array_equal(model.predict([[9]]), ['cat'])

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(2, 'unknown')
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(2).fit(np.empty((0, 2)), [])
