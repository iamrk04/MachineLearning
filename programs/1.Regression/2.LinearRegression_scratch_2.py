"""
Linear Regression for housing price prediction.
No external libraries. End-to-end working code.

Model:  y = Xw + b
Loss:   J = (1/n) * Σ(y_pred - y)²
Grads:  dw = (2/n) * Xᵀ(y_pred - y),  db = (2/n) * Σ(y_pred - y)
"""


class LinearRegression:
    """Multivariate linear regression with gradient descent."""

    def __init__(self, n_iters: int = 10_000, lr: float = 0.1) -> None:
        self.weights: list[float] = []
        self.bias: float = 0.0
        self.n_iters = n_iters
        self.lr = lr

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _dot_row(row: list[float], weights: list[float], bias: float) -> float:
        """Compute w·x + b for a single sample using sum + zip (faster than index loops)."""
        return sum(x * w for x, w in zip(row, weights)) + bias

    # ── core API ─────────────────────────────────────────────────────────

    def fit(self, X: list[list[float]], y: list[float]) -> "LinearRegression":
        n_samples = len(X)
        n_feat = len(X[0])
        inv_n = 1.0 / n_samples

        self.weights = [0.0] * n_feat
        self.bias = 0.0

        for _ in range(self.n_iters):
            # Forward pass – compute predictions
            y_pred = [self._dot_row(row, self.weights, self.bias) for row in X]

            # Residuals scaled by 1/n
            residuals = [inv_n * (yp - yt) for yp, yt in zip(y_pred, y)]

            # Gradients
            db = sum(residuals)
            dw = [
                sum(r * row[j] for r, row in zip(residuals, X))
                for j in range(n_feat)
            ]

            # Parameter update
            self.weights = [w - self.lr * g for w, g in zip(self.weights, dw)]
            self.bias -= self.lr * db

        return self

    def predict(self, X: list[list[float]]) -> list[float]:
        return [self._dot_row(row, self.weights, self.bias) for row in X]

    def mse(self, X: list[list[float]], y: list[float]) -> float:
        """Mean squared error on the given data."""
        preds = self.predict(X)
        return sum((p - t) ** 2 for p, t in zip(preds, y)) / len(y)


# ── Data ─────────────────────────────────────────────────────────────────

# [sqft, num_rooms, neighborhood_safety]
X = [
    [1500, 3, 5],
    [2000, 3, 7],
    [1200, 2, 4],
    [2800, 4, 8],
    [1800, 3, 6],
    [2200, 4, 5],
    [1100, 2, 3],
    [3500, 5, 9],
]
y = [250_000, 315_000, 190_000, 450_000, 280_000, 310_000, 165_000, 590_000]


# ── Min-max normalisation ────────────────────────────────────────────────

def minmax_normalize(data: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    """Compute and apply min-max normalisation. Returns (normalised_data, mins, maxs)."""
    n_feat = len(data[0])
    mins = [min(row[j] for row in data) for j in range(n_feat)]
    maxs = [max(row[j] for row in data) for j in range(n_feat)]
    normed = [
        [(row[j] - mins[j]) / (maxs[j] - mins[j]) for j in range(n_feat)]
        for row in data
    ]
    return normed, mins, maxs


def apply_normalize(data: list[list[float]], mins: list[float], maxs: list[float]) -> list[list[float]]:
    """Apply pre-computed min-max normalisation."""
    n_feat = len(data[0])
    return [
        [(row[j] - mins[j]) / (maxs[j] - mins[j]) for j in range(n_feat)]
        for row in data
    ]


# ── Train / test split & normalisation ───────────────────────────────────

X_train_raw, X_test_raw = X[:-1], [X[-1]]
y_train, y_test = y[:-1], [y[-1]]

# Fit normalisation on training data only (avoid data leakage)
X_train, mins, maxs = minmax_normalize(X_train_raw)
X_test = apply_normalize(X_test_raw, mins, maxs)

# ── Train & evaluate ─────────────────────────────────────────────────────

model = LinearRegression(n_iters=10_000, lr=0.1)
model.fit(X_train, y_train)

prediction = model.predict(X_test)
print(f"Prediction : {prediction[0]:,.0f}")
print(f"Actual     : {y_test[0]:,.0f}")
print(f"Train MSE  : {model.mse(X_train, y_train):,.0f}")