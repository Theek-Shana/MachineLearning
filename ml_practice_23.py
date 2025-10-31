# ml_practice_23.py
# Comparing K-Means vs DBSCAN visually

from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt

# 1️⃣ Create sample data (two interlocking moons)
X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

# 2️⃣ Apply K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# 3️⃣ Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.2, min_samples=5)
y_dbscan = dbscan.fit_predict(X)

# 4️⃣ Plot the comparison side-by-side
plt.figure(figsize=(12, 5))

# K-Means plot
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap="viridis", s=50)
plt.title("K-Means Clustering (Fails on Moons)")

# DBSCAN plot
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap="viridis", s=50)
plt.title("DBSCAN Clustering (Captures Moons)")

plt.show()
