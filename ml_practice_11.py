# 11.py - Using Regularization (Ridge & Lasso) to Reduce Overfitting

from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Generate noisy nonlinear data
np.random.seed(42)
X = 3 * np.random.rand(100, 1) - 1.5
y = 0.5 * X**3 - X**2 + 2 * X + np.random.randn(100, 1) * 2
y = y.ravel()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 2: Create Polynomial Regression (degree=15 — overfits easily)
poly_degree = 15

# Ridge Regression (L2 regularization)
ridge_model = make_pipeline(PolynomialFeatures(poly_degree), Ridge(alpha=1.0))
ridge_model.fit(X_train, y_train)

# Lasso Regression (L1 regularization)
lasso_model = make_pipeline(PolynomialFeatures(poly_degree), Lasso(alpha=0.001, max_iter=10000))
lasso_model.fit(X_train, y_train)

# Step 3: Evaluate and compare
y_pred_ridge = ridge_model.predict(X_test)
y_pred_lasso = lasso_model.predict(X_test)

ridge_mse = mean_squared_error(y_test, y_pred_ridge)
lasso_mse = mean_squared_error(y_test, y_pred_lasso)

print("Ridge Regression Test MSE:", ridge_mse)
print("Lasso Regression Test MSE:", lasso_mse)

# Step 4: Plot comparison
X_fit = np.linspace(-1.5, 1.5, 200).reshape(-1, 1)
y_fit_ridge = ridge_model.predict(X_fit)
y_fit_lasso = lasso_model.predict(X_fit)

plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, color='lightgray', label='Train Data')
plt.plot(X_fit, y_fit_ridge, color='blue', label='Ridge Model')
plt.plot(X_fit, y_fit_lasso, color='red', linestyle='--', label='Lasso Model')
plt.legend()
plt.title("Ridge vs Lasso Regularization (Degree 15 Polynomial)")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
