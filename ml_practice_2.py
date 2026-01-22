# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 2: Create simple dataset (study hours vs marks)
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Marks': [20, 25, 35, 45, 50, 60, 65, 70, 80, 85]
}
df = pd.DataFrame(data)
 
# Step 3: Split input (X) and output (y)
X = df[['Hours']]
y = df['Marks']

# Step 4: Split for training & testing (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Step 5: Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict test data
y_pred = model.predict(X_test)

# Step 7: Evaluate accuracy
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Step 8: Try your own prediction
hours = float(input("Enter study hours: "))
predicted_marks = model.predict([[hours]])
print(f"Predicted Marks for {hours} hours = {predicted_marks[0]:.2f}")


