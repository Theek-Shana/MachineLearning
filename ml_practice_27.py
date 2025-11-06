# ml_practice_27.py
# Train/Test split and evaluation on Iris dataset

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1️⃣ Load dataset
iris = load_iris()
X = iris.data[:, (2, 3)]   # petal length & width
y = iris.target            # 3 classes: Setosa, Versicolor, Virginica

# 2️⃣ Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3️⃣ Train Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 4️⃣ Predict on Test Data
y_pred = model.predict(X_test)

# 5️⃣ Evaluate Performance
accuracy = accuracy_score(y_test, y_pred)
print("✅ Model accuracy:", accuracy)

# Test on custom new flower
test_sample = [[5.1, 1.8]]
pred = model.predict(test_sample)
print(f"Prediction for {test_sample}: {iris.target_names[pred][0]}")
