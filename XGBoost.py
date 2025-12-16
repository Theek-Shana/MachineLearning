import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Sample data
X = pd.DataFrame({
    'temp': [30, 32, 28, 35, 40, 22],
    'humidity': [70, 65, 80, 60, 55, 90]
})

y = [25, 27, 23, 30, 35, 20]

# Train / validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# XGBoost model
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)

# Train
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# Predict
preds = model.predict(X_val)

# Evaluate
mae = mean_absolute_error(y_val, preds)
print("MAE:", mae)
