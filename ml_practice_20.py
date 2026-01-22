# ml_practice_20.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate some random data
np.random.seed(42)
X = np.random.randn(200, 2)

# Apply K-Means clustering

kmeans = KMeans(n_clusters=3, random_state=42)
y_pred = kmeans.fit_predict(X)

# Plot the clusters  
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="viridis", s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c="red", marker="X", s=200, label="Centroids")
plt.title("K-Means Clustering Example")
plt.legend() 

plt.show()
 





