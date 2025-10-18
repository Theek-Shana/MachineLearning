# ml_practice_13.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Generate synthetic data (same as before)
np.random.seed(42)
X = np.random.rand(100, 1) * 4 - 2
y = 0.5 * X**3 - X**2 + X + np.random.randn(100, 1) * 0.8

# Cross-validation setup (5 folds)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

degrees = range(1, 11)
mean_scores = []

for degree in degrees:
    # Build pipeline: Polynomial -> Linear Regression
    model = Pipeline([
        ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
        ("lin_reg", LinearRegression())
    ])
    
    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
    mean_score = np.mean(scores)
    mean_scores.append(mean_score)

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(degrees, mean_scores, "o-", color="blue")
plt.title("Model Performance vs Polynomial Degree")
plt.xlabel("Polynomial Degree")
plt.ylabel("Negative Mean Squared Error (Higher = Better)")
plt.grid(True)
plt.show()
