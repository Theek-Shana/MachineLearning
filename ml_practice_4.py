# ml_practice_4.py

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

# 1. Load the California housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# 2. Show first few rows
print("First 5 rows of the dataset:")
print(df.head(), "\n")

# 3. Split into train/test
X = df.drop("MedHouseVal", axis=1)  # features
y = df["MedHouseVal"]               # target (median house value)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Test model performance
score = model.score(X_test, y_test)
print(f"Model R² score: {score:.3f}")

# 6. Make predictions
predictions = model.predict(X_test[:5])
print("\nPredictions for first 5 houses:", predictions)
print("Actual values:", list(y_test[:5]))
