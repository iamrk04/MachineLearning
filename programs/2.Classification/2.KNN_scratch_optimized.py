from collections import Counter
from bisect import bisect_left

import numpy as np


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:

    def __init__(self, k):
        self.k = k
    
    def fit(self, X, y):
        # Sort the training data based on the first feature
        sorted_indices = np.argsort(X[:, 0])
        self.X_train = X[sorted_indices]
        self.y_train = y[sorted_indices]
    
    def predict(self, X):
        pred = [self._predict(x) for x in X]
        return np.array(pred)
    
    def _predict(self, x):
        # Use binary search to find the position to insert x
        idx = bisect_left(self.X_train[:, 0], x[0])
        
        # Get k nearest neighbors
        left = max(0, idx - self.k)
        right = min(len(self.X_train), idx + self.k)
        
        neighbors = []
        for i in range(left, right):
            neighbors.append((self.euclidean_distance(x, self.X_train[i]), self.y_train[i]))
        
        # Sort neighbors by distance and get the k closest ones
        neighbors.sort(key=lambda x: x[0])
        k_nearest = [neighbor[1] for neighbor in neighbors[:self.k]]
        
        return Counter(k_nearest).most_common(1)[0][0]


X_train = np.array([
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3],
    [4, 4, 4, 4],
    [5, 5, 5, 5]
])
y_train = np.array([1, 2, 2, 2, 2])
X_test = np.array([[2.5, 2.5, 2.5, 2.5]])

model = KNN(k=3)
model.fit(X_train, y_train)
print(model.predict(X_test))
