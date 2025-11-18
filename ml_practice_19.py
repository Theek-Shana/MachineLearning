# ml_practice_19.py
# Visualizing Overfitting & Underfitting with Learning Curves

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Function to plot learning curves
def plot_learning_curves(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring="neg_mean_squared_error",
        train_sizes=np.linspace(0.1, 1.0, 20), random_state=42
    )
    
    train_errors = -train_scores.mean(axis=1)
    test_errors = -test_scores.mean(axis=1) 

    plt.plot(train_sizes, np.sqrt(train_errors), "o-", label="Training error")
    plt.plot(train_sizes, np.sqrt(test_errors), "o-", label="Testing error")
    plt.xlabel("Training set size")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True)
    plt.show()

# Generate some data
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

# Try linear model (underfitting)
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)

# Try polynomial model (better fit)
poly_reg = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(poly_reg, X, y)

