# ml_practice_25.py
# Applying DBSCAN on the real-world Iris dataset

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np

# 1️⃣ Load the Iris dataset
iris = load_iris()
X = iris.data[:, (2, 3)]  # use only petal length & petal width

# 2️⃣ Normalize the data (important for DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3️⃣ Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
y_pred = dbscan.fit_predict(X_scaled)

# 4️⃣ Plot results
plt.figure(figsize=(7, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred, cmap="viridis", s=60)
plt.title("DBSCAN Clustering on Iris Dataset (Petal Length vs Width)")
plt.xlabel("Petal length (standardized)")
plt.ylabel("Petal width (standardized)")
plt.show()

# 5️⃣ Print cluster labels
print("Cluster labels:", np.unique(y_pred))
print("Number of noise points:", np.sum(y_pred == -1))
