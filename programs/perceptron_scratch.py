# y_pred = X * w + b
# w = w - lr * dw = w - lr * 1/n_samples * (y_pred - y) * X
# b = b - lr * db = b - lr * 1/n_samples * (y_pred - y)
# J = 1 / n_samples * (y_pred - y) ** 2

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Perceptron:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.wt = None
        self.bias = None
        self.act_fn = sigmoid

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.wt = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = self.act_fn(np.dot(X, self.wt) + self.bias)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.wt -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_pred = self.act_fn(np.dot(X, self.wt) + self.bias)
        return y_pred
