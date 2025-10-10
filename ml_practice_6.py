# ml_practice_6.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor

# 1. Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# 2. Use only Median Income as feature
X = df[["MedInc"]]
y = df["MedHouseVal"]

# 3. Train Decision Tree model
tree_reg = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_reg.fit(X, y)

# 4. Create new data for prediction (for smooth curve)
X_new = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = tree_reg.predict(X_new)

# 5. Plot results
plt.figure(figsize=(8, 6))
plt.scatter(X, y, alpha=0.3, color="gray", label="Actual Data")
plt.plot(X_new, y_pred, color="green", linewidth=2, label="Decision Tree Prediction")
plt.xlabel("Median Income (×$10,000)")
plt.ylabel("Median House Value (×$100,000)")
plt.title("Decision Tree Regression (max_depth=3)")
plt.legend()
plt.grid(True)
plt.show()
