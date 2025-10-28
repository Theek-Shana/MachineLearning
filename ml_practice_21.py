# ml_practice_21.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate random 2D data
np.random.seed(42)
X = np.random.randn(200, 2)

# Compute inertia for different K values
inertias = []
k_values = range(1, 10)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.plot(k_values, inertias, "bo-")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()
