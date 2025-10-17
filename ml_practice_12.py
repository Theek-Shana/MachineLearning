# 12.py - Visualizing Bias vs Variance (Training vs Validation Error)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate nonlinear data (a smooth curve + noise)
np.random.seed(42)
X = np.random.rand(100, 1) * 4 - 2  # range -2 to 2
y = 0.5 * X**3 - X**2 + X + np.random.randn(100, 1) * 0.8
y = y.ravel()

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 2: Prepare containers for errors
degrees = range(1, 15)
train_errors, test_errors = [], []

# Step 3: Train models with different polynomial degrees
for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))

# Step 4: Plot training vs test errors
plt.figure(figsize=(8, 6))
plt.plot(degrees, train_errors, "o-", label="Training Error")
plt.plot(degrees, test_errors, "s-", label="Validation Error")
plt.xlabel("Model Complexity (Polynomial Degree)")
plt.ylabel("Mean Squared Error")
plt.title("Bias vs Variance Tradeoff")
plt.legend()
plt.grid(True)
plt.show()
