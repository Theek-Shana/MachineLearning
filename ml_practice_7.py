# ml_practice_7.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.datasets import fetch_california_housing

# Load California housing data
housing = fetch_california_housing(as_frame=True)
X = housing.data[["MedInc"]]   # Median income
y = housing.target             # Median house value

# Train a Decision Tree Regressor
tree_reg = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_reg.fit(X, y)

# Visualize the decision tree structure
plt.figure(figsize=(16, 10))
plot_tree(tree_reg, feature_names=["MedInc"], filled=True, rounded=True)
plt.title("Decision Tree Regression Structure (Depth=3)", fontsize=16)
plt.show()
