from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1️⃣ Load MNIST
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist["data"], mnist["target"].astype(int)

# 2️⃣ Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3️⃣ Create Random Forest
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 4️⃣ Train the forest
rnd_clf.fit(X_train, y_train)

# 5️⃣ Make predictions
y_pred = rnd_clf.predict(X_test)

# 6️⃣ Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest accuracy on MNIST: {accuracy:.4f}")

# 7️⃣ Optional: predict a single digit
import matplotlib.pyplot as plt
import numpy as np

some_index = 0
image = X_test[some_index].reshape(28, 28)
plt.imshow(image, cmap="gray")
plt.title(f"Predicted: {rnd_clf.predict([X_test[some_index]])[0]}")
plt.axis("off")
plt.show()
