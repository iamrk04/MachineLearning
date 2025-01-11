# w = w - lr * dw
# b = b - lr * db
# y_pred = w * x + b
# J = 1 / n_samples * sum((y_pred - y) ** 2)
# dw = 1 / n_samples * (y_pred - y) * x
# db = 1 / n_samples * (y_pred - y)


import numpy as np


class LinearModel:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.wt = None
        self.bias = None

    def activation_fn(self, x):
        raise NotImplementedError

    # TC: O(n_iters * n_samples * n_features)
    # SC: O(n_features + n_samples)
    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.wt = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = self.activation_fn(np.dot(X, self.wt) + self.bias)
            dw = (1 / n_samples) * np.dot(X.T, y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.wt -= self.lr * dw
            self.wt -= self.lr * db

    # TC: O(n_samples * n_features)
    # SC: O(n_samples)
    def predict(self, X):
        y_pred = self.activation_fn(np.dot(X, self.wt) + self.bias)
        return y_pred


class LinearRegression(LinearModel):

    def activation_fn(self, x):
        return x


class LogisticRegression(LinearModel):

    def activation_fn(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, X):
        y_prob = super().predict(X)
        return np.array([1 if prob > 0.5 else 0 for prob in y_prob])


if __name__ == "__main__":
    # linear regression
    X_train = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    y_train = np.array([1, 2, 3, 4])
    X_test = np.array([[5, 5], [6, 6]])
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred = lin_reg.predict(X_test)  # [5, 6]
    print(y_pred)

    # logistic regression
    X_train = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[0, 0], [6, 6]])
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)  # [0, 1]
    print(y_pred)
