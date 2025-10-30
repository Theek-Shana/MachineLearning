# ml_practice_22.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# Generate 2D moon-shaped dataset
X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
y_pred = dbscan.fit_predict(X)

# Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="viridis", s=50)
plt.title("DBSCAN Clustering")
plt.show()
