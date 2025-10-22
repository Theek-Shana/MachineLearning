# ml_practice_17.py
# Underfitting vs Good Fit vs Overfitting (Polynomial Regression)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Generate data
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

# Create polynomial models with different degrees
degrees = [1, 2, 15]
plt.figure(figsize=(12, 4))

for i, degree in enumerate(degrees, 1):
    model = Pipeline([
        ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
        ("lin_reg", LinearRegression()),
    ])
    model.fit(X, y)

    X_new = np.linspace(-3, 3, 100).reshape(100, 1)
    y_new = model.predict(X_new)

    plt.subplot(1, 3, i)
    plt.plot(X, y, "b.", label="Training data")
    plt.plot(X_new, y_new, "r-", linewidth=2, label=f"degree={degree}")
    plt.title(f"Polynomial degree = {degree}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

plt.suptitle("Underfitting vs Good Fit vs Overfitting", fontsize=14)
plt.tight_layout()
plt.show()
