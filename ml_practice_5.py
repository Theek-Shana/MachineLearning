# ml_practice_5.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

# 1. Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# 2. Choose one feature: Median Income (MedInc)
X = df[["MedInc"]]
y = df["MedHouseVal"]

# 3. Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# 4. Predict values
X_new = pd.DataFrame({"MedInc": [1.5, 3, 4.5, 6, 7.5, 9]})
y_pred = model.predict(X_new)

# 5. Plot
plt.figure(figsize=(8, 6))
plt.scatter(X, y, alpha=0.3, label="Actual Data", color="gray")
plt.plot(X_new, y_pred, color="red", linewidth=2, label="Regression Line")
plt.xlabel("Median Income (×$10,000)")
plt.ylabel("Median House Value (×$100,000)")
plt.title("Linear Relationship between Income and House Value")
plt.legend()
plt.grid(True)
plt.show()
