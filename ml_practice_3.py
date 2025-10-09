# Step 1: Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# Step 2: Load the data
# This CSV is available in the GitHub repo data folder
data_url = "https://github.com/ageron/data/raw/main/lifesat/lifesat.csv"
lifesat = pd.read_csv(data_url)

# Step 3: Prepare the data
X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

# Step 4: Visualize the data
plt.scatter(X, y)
plt.xlabel("GDP per capita (USD)")
plt.ylabel("Life Satisfaction")
plt.title("Life Satisfaction vs GDP per Capita")
plt.grid(True)
plt.show()

# Step 5: Train a Linear Regression model
lin_model = LinearRegression()
lin_model.fit(X, y)

# Step 6: Train a K-Nearest Neighbors (KNN) model
knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X, y)

# Step 7: Predict life satisfaction for a new country GDP
X_new = [[37_655.2]]  # Example GDP per capita
lin_pred = lin_model.predict(X_new)
knn_pred = knn_model.predict(X_new)

print(f"Linear Regression Prediction: {lin_pred[0][0]:.2f}")
print(f"KNN (k=3) Prediction: {knn_pred[0][0]:.2f}")

# Step 8: Visualize both models
X_fit = np.linspace(23_500, 62_500, 1000).reshape(-1, 1)
y_lin = lin_model.predict(X_fit)
y_knn = knn_model.predict(X_fit)

plt.plot(X, y, "b.", label="Data points")
plt.plot(X_fit, y_lin, "r-", label="Linear Regression")
plt.plot(X_fit, y_knn, "g--", label="KNN (k=3)")
plt.xlabel("GDP per capita (USD)")
plt.ylabel("Life Satisfaction")
plt.legend()
plt.grid(True)
plt.title("Model Comparison: Linear Regression vs KNN")
plt.show()
