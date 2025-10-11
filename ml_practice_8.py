# ml_practice_8.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing

# Load the California housing dataset
housing = fetch_california_housing(as_frame=True)
X = housing.data[["MedInc"]]   # Median income
y = housing.target             # Median house price

# Sort the data for a smoother plot
X_sorted_idx = np.argsort(X.values.flatten())
X_sorted = X.values[X_sorted_idx]
y_sorted = y.values[X_sorted_idx]

# Train Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Train Decision Tree Regressor
tree_reg = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_reg.fit(X, y)

# Make predictions
X_new = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
y_lin_pred = lin_reg.predict(X_new)
y_tree_pred = tree_reg.predict(X_new)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y, s=10, label="Data points", alpha=0.4)
plt.plot(X_new, y_lin_pred, "r-", linewidth=2, label="Linear Regression")
plt.plot(X_new, y_tree_pred, "g-", linewidth=2, label="Decision Tree (Depth=3)")
plt.xlabel("Median Income (MedInc)")
plt.ylabel("Median House Value")
plt.title("Linear Regression vs Decision Tree Regression")
plt.legend()
plt.grid(True)
plt.show()
