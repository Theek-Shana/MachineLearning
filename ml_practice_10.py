# 10.py - Demonstrating Overfitting and How to Fix It

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Create sample regression data
X, y = make_regression(n_samples=40, n_features=1, noise=15, random_state=42) 

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 2: Create Polynomial features (high degree causes overfitting)
poly_high = PolynomialFeatures(degree=15)
X_train_high = poly_high.fit_transform(X_train) 
X_test_high = poly_high.transform(X_test)

# Fit model (overfitting example)
model_high = LinearRegression()
model_high.fit(X_train_high, y_train)

# Predict
y_pred_train = model_high.predict(X_train_high)
y_pred_test = model_high.predict(X_test_high)

# Step 3: Evaluate
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

print("Training MSE:", train_mse)
print("Testing MSE:", test_mse)
print("\nNotice: The training error is small but test error is large -> Overfitting!")

# Step 4: Plot to visualize
X_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_fit_poly = poly_high.transform(X_fit)
y_fit = model_high.predict(X_fit_poly)

plt.scatter(X_train, y_train, color='blue', label='Train Data')
plt.scatter(X_test, y_test, color='orange', label='Test Data')
plt.plot(X_fit, y_fit, color='red', label='Overfitted Model')
plt.legend()
plt.title("Overfitting Example - High Polynomial Degree")
plt.show()

# Step 5: Fix - reduce degree to avoid overfitting
poly_low = PolynomialFeatures(degree=2)
X_train_low = poly_low.fit_transform(X_train)
X_test_low = poly_low.transform(X_test)

model_low = LinearRegression()
model_low.fit(X_train_low, y_train)

y_pred_low = model_low.predict(X_test_low)
fixed_mse = mean_squared_error(y_test, y_pred_low)
print("\nAfter reducing degree to 2, Test MSE:", fixed_mse)


