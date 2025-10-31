# ml_practice_24.py
# Detecting and removing noise (outliers) using DBSCAN

from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np

# 1️⃣ Generate moon-shaped data with more noise
X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

# 2️⃣ Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.2, min_samples=5)
y_pred = dbscan.fit_predict(X)

# 3️⃣ Identify noise points (DBSCAN labels them as -1)
core_mask = (y_pred != -1)   # True for core/border points
noise_mask = (y_pred == -1)  # True for noise points

# 4️⃣ Plot clusters and highlight noise
plt.figure(figsize=(7, 5))
plt.scatter(X[core_mask, 0], X[core_mask, 1], c=y_pred[core_mask], cmap="viridis", s=50, label="Clustered Points")
plt.scatter(X[noise_mask, 0], X[noise_mask, 1], c="red", marker="x", s=70, label="Noise (Outliers)")
plt.title("DBSCAN – Detecting and Removing Noise")
plt.legend()
plt.show()
