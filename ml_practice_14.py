# ml_practice_14.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Generate data (nonlinear relation)
np.random.seed(42)
X = np.random.rand(100, 1) * 4 - 2
y = 0.5 * X**3 - X**2 + X + np.random.randn(100, 1) * 0.8

# Step 2: Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the model (Linear Regression) 
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")

# Step 6: Visualize predictions vs real data
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted', marker='x')
plt.xlabel("X value")
plt.ylabel("Target (y)")
plt.title("Model Predictions vs Actual Values")
plt.legend()
plt.grid(True)
plt.show()

