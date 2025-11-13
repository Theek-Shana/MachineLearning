# ml_practice_31.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1️⃣ Standardization (Z-Score)
scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)
X_test_std = scaler_std.transform(X_test)

# 2️⃣ Min-Max Normalization
scaler_mm = MinMaxScaler()
X_train_mm = scaler_mm.fit_transform(X_train)
X_test_mm = scaler_mm.transform(X_test)

# 3️⃣ Train model with standardized data
model_std = LogisticRegression(max_iter=1000)
model_std.fit(X_train_std, y_train)
y_pred_std = model_std.predict(X_test_std)

# 4️⃣ Train model with normalized data
model_mm = LogisticRegression(max_iter=1000)
model_mm.fit(X_train_mm, y_train)
y_pred_mm = model_mm.predict(X_test_mm)

# 5️⃣ Compare Accuracy
acc_std = accuracy_score(y_test, y_pred_std)
acc_mm = accuracy_score(y_test, y_pred_mm)

print("StandardScaler Accuracy:", acc_std)
print("MinMaxScaler Accuracy:", acc_mm)
