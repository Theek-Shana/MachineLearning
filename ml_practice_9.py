# ml_practice_9.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing

# Load California housing data
housing = fetch_california_housing(as_frame=True)
X = housing.data[["MedInc"]]
y = housing.target

# Prepare a smooth range for predictions
X_new = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)

# Train two models with different depths
tree_shallow = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_deep = DecisionTreeRegressor(max_depth=None, random_state=42)  # full depth (no limit)

tree_shallow.fit(X, y)
tree_deep.fit(X, y)

# Predictions
y_pred_shallow = tree_shallow.predict(X_new)
y_pred_deep = tree_deep.predict(X_new)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, s=10, alpha=0.3, label="Training Data")
plt.plot(X_new, y_pred_shallow, "r-", linewidth=2, label="Tree (max_depth=3)")
plt.plot(X_new, y_pred_deep, "g-", linewidth=2, label="Tree (no limit)")
plt.xlabel("Median Income (MedInc)")
plt.ylabel("Median House Value")
plt.title("Overfitting Example: Decision Tree Depth")
plt.legend()
plt.grid(True)
plt.show()
