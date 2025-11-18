# ml_practice_16.py
# Polynomial Regression vs Linear Regression (continued)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Generate dataset
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3 
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

# Two pipelines: one linear, one polynomial degree 10
lin_reg = LinearRegression()
poly_reg = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
lin_reg.fit(X, y)
poly_reg.fit(X, y)

# Plot predictions
X_new = np.linspace(-3, 3, 100).reshape(100, 1)
y_lin_pred = lin_reg.predict(X_new)
y_poly_pred = poly_reg.predict(X_new)

plt.figure(figsize=(8, 5))
plt.plot(X, y, "b.", label="Training data")
plt.plot(X_new, y_lin_pred, "g-", linewidth=2, label="Linear Regression")
plt.plot(X_new, y_poly_pred, "r--", linewidth=2, label="Polynomial (degree=10)")
plt.legend(loc="upper left")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear vs Polynomial Regression (degree=10)")
plt.grid(True)
plt.show()

