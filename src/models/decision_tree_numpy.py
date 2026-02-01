import numpy as np

class DecisionTreeNumpy:
    """
    A simple implementation of a Decision Tree for classification using NumPy only.
    """

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Decision Tree model.
        Args:
            X (np.ndarray): The input features, shape (n_samples, n_features).
            y (np.ndarray): The target values, shape (n_samples,).
        """
        self.tree = self._build_tree(X, y, depth=0)

    def _gini(self, y:np.ndarray) -> float:
        """
        Docstring for _gini
        Gini is a measure of statistical dispersion that represents the inequality among values of a frequency distribution.
        It is commonly used as a metric for evaluating the quality of splits in decision trees.
        Gini = 1 - sum(p_i)^2
        where p_i is the proportion of instances belonging to class i.

        :param self: class instance
        :type self: DecisionTreeNumpy
        :param y: class labels
        :type y: np.ndarray
        :return: Gini impurity
        :rtype: float
        """
        
        _, class_counts = np.unique(y, return_counts=True)
        total_samples = y.shape[0]
        total = 0
        for count in class_counts:
            prob = count / total_samples
            total += prob ** 2
        gini = 1.0 - total
        return gini

    def _find_best_split(self, X:np.ndarray, y:np.ndarray) -> tuple:
        """
        this methods finds the best feature and threshold to split on based on Gini impurity.
        
        :param self: class instance
        :param X: training features
        :type X: np.ndarray
        :param y: labels
        :type y: np.ndarray
        :return: the best feature index and threshold to split on
        :rtype: tuple
        """
        n_samples, n_features = X.shape
        best_gini = float("inf")
        best_feature_index = None
        best_threshold = None

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index]) # gets all possible unique values for given feature
            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold # gets everyone who is less than the current threshold
                right_indices = X[:, feature_index] > threshold # gets the rest

                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue # not a good threshold - one side has no samples

                gini_left = self._gini(y[left_indices])
                gini_right = self._gini(y[right_indices])
                weighted_gini = (
                    len(y[left_indices]) * gini_left + len(y[right_indices]) * gini_right
                ) / n_samples

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold


    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> dict:
        """
        Docstring for _build_tree
        
        :param self: Description
        :param X: Description
        :type X: np.ndarray
        :param y: Description
        :type y: np.ndarray
        :return: Description
        :rtype: dict
        """
        n_samples = X.shape[0]
        unique_classes, class_counts = np.unique(y, return_counts=True)

        # Stop conditions
        if depth is None:
            raise ValueError("Depth must be provided for building the tree.")
        
        if depth >= self.max_depth or \
            n_samples < self.min_samples_split or \
            len(unique_classes) == 1:

            majority_class = unique_classes[np.argmax(class_counts)]
            return {"type": "leaf", "class": majority_class}

        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            majority_class = unique_classes[np.argmax(class_counts)]
            return {"type": "leaf", "class": majority_class}
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            "type": "node",
            "feature_index": best_feature,
            "threshold": best_threshold,
            "left": left_child,
            "right": right_child
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained Decision Tree model.
        Args:
            X (np.ndarray): The input features, shape (n_samples, n_features).
        Returns:
            np.ndarray: The predicted class labels, shape (n_samples,).
        """
        predictions = np.array([self._predict_sample(sample, self.tree) for sample in X])
        return predictions

    def _predict_sample(self, sample: np.ndarray, tree: dict) -> int:
        """
        Predict the class label for a single sample.
        Args:
            sample (np.ndarray): A single input sample, shape (n_features,).
            tree (dict): The decision tree.
        Returns:
            int: The predicted class label.
        """
        if tree["type"] == "leaf":
            return tree["class"]

        feature_value = sample[tree["feature_index"]]
        if feature_value <= tree["threshold"]:
            return self._predict_sample(sample, tree["left"])
        else:
            return self._predict_sample(sample, tree["right"])

    def print_tree(self) -> None:
        """
        Prints the decision tree structure.
        """
        node = self.tree
        if node is None:
            print("The tree has not been trained yet.")
            return
        
        self._print_tree_recursive(node)
    
    def _print_tree_recursive(self, node: dict, depth: int = 0) -> None:
        indent = "  " * depth
        if node["type"] == "leaf":
            print(f"{indent}Leaf: Class={node['class']}")
        else:
            print(f"{indent}Node: Feature {node['feature_index']} <= {node['threshold']}")
            self._print_tree_recursive(node["left"], depth + 1)
            self._print_tree_recursive(node["right"], depth + 1)
        