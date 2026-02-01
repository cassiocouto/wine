"""
File containing a class for Linear Regression using NumPy.
For learning purposes only.
Linear Regression models are better implemented using libraries like scikit-learn.
"""

import numpy as np


class LinearRegressionNumpy:
    """
    A simple implementation of Linear Regression using NumPy.
    """

    def __init__(self):
        self.weights = None

    def fit(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.01, n_iterations: int = 1000) -> None:
        """
        Train the model using the Normal Equation.
        Steps:
        1. Add a bias term (column of 1s) to X
        2. Apply the formula: w = (X^T * X)^(-1) * X^T * y
        3. Retrain the weights at the n_interations specified
        Args:
            X (np.ndarray): The input features, shape (n_samples, n_features).
            y (np.ndarray): The target values, shape (n_samples,).
        """
        
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        n_samples, n_features = X_b.shape

        self.weights = np.zeros(n_features)

        for _ in range(n_iterations):
            y_pred = X_b @ self.weights
            error = y_pred - y
            gradient = (2 /n_samples) * (X_b.T @ error)
            self.weights -= learning_rate * gradient
        

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        Steps:
        1. Add a bias term (column of 1s) to X
        2. Compute predictions: y_pred = X_b * w
        Args:
            X (np.ndarray): The input features, shape (n_samples, n_features).
        Returns:
            np.ndarray: The predicted values, shape (n_samples,).
        """
        if self.weights is None:
            raise ValueError(
                "Model is not trained yet. Please call 'fit' before 'predict'."
            )

        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        y_predict = X_b.dot(self.weights)
        return y_predict
    
    def mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Mean Squared Error between true and predicted values.
        Args:
            y_true (np.ndarray): The true target values, shape (n_samples,).
            y_pred (np.ndarray): The predicted target values, shape (n_samples,).
        Returns:
            float: The Mean Squared Error.
        """
        return np.mean((y_true - y_pred) ** 2)


if __name__ == "__main__":
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1.5, 1.7, 2.9, 4, 5])

    model = LinearRegressionNumpy()
    model.fit(X, y)
    X_new = np.array([[100], [101]])
    predictions = model.predict(X_new)
    print("Predictions:", predictions)
    print("MSE on training data:", model.mean_squared_error(y, model.predict(X)))
