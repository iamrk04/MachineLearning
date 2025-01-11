import numpy as np


class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    # TC: O(n_features**2 * n_samples + n_features**3 + n_features log (n_features))
    # SC: O(n_features ** 2)
    def fit(self, X):
        # center the data
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # compute covariance matrix
        cov = np.cov(X.T)

        # compute eigenvalue and eigenmatrix
        eigen_values, eigen_vectors = np.linalg.eig(cov)
        eigen_vectors = eigen_vectors.T

        # sort the eigen value
        idxs = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[idxs]
        eigen_vectors = eigen_vectors[idxs]

        # store first n eigen vectors
        self.components = eigen_vectors[: self.n_components]
    
    # TC: O(n_samples * n_features + n_samples * n_features * k)
    # SC : O(n_samples * k)
    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)


if __name__ == "__main__":
    X = np.array([
        [126, 78], [128, 80], [128, 82], [130, 82], [130, 84], [132, 86]
    ])
    pca = PCA(1)
    pca.fit(X)
    pr_comp = pca.transform(X)
    print(pr_comp)
