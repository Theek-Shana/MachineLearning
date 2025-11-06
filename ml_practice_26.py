# ml_practice_26.py
# Iris flower classification using Logistic Regression

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

# 1️⃣ Load dataset
iris = load_iris()
X = iris.data[:, (2, 3)]  # petal length & petal width
y = (iris.target == 2).astype(int)  # classify "Virginica vs Not Virginica"

# 2️⃣ Train model
model = LogisticRegression()
model.fit(X, y)

# 3️⃣ Test prediction
test = [[5.0, 1.8]]  # petal length & width
prediction = model.predict(test)
print(f"Prediction for {test}: {'Virginica' if prediction[0] else 'Not Virginica'}")

# 4️⃣ Plot decision boundary
x0 = np.linspace(2, 7, 100)
x1 = np.linspace(0, 3, 100)
xx, yy = np.meshgrid(x0, x1)
grid = np.c_[xx.ravel(), yy.ravel()]
probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

plt.contourf(xx, yy, probs, 25, alpha=0.7)
plt.scatter(X[:,0], X[:,1], c=y, cmap="viridis", edgecolors="k")
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.title("Logistic Regression - Virginica Classification")
plt.colorbar(label="Probability Virginica")
plt.show()
