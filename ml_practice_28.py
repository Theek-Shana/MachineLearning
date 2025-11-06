# ml_practice_28.py
# Decision Tree Classifier on Iris dataset

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = iris.data[:, (2, 3)]  # petal length and width
y = iris.target           # class labels

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Decision Tree
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("✅ Decision Tree Accuracy:", acc)

# Visualize the tree
plt.figure(figsize=(10,6))
plot_tree(model, feature_names=["petal length", "petal width"],
          class_names=iris.target_names, filled=True)
plt.title("Decision Tree for Iris Classification")
plt.show()
