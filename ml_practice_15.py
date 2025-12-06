import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Generate data
np.random.seed(42)
X = np.linspace(0, 3, 30).reshape(-1, 1)
y = np.sin(X) + np.random.randn(30, 1) * 0.3

# Polynomial degree
degree = 15

# Ridge Regression (L2 Regularization)
ridge_model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1.0)) 
ridge_model.fit(X, y)
y_ridge = ridge_model.predict(X)

# Lasso Regression (L1 Regularization)
lasso_model = make_pipeline(PolynomialFeatures(degree), Lasso(alpha=0.01, max_iter=10000))
lasso_model.fit(X, y)
y_lasso = lasso_model.predict(X)

# Plot comparison
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='black', label='Data')
plt.plot(X, y_ridge, color='blue', label='Ridge Regression')
plt.plot(X, y_lasso, color='red', label='Lasso Regression')
plt.legend()
plt.title("Ridge vs Lasso Regression (Polynomial Degree = 15)")
plt.show()

