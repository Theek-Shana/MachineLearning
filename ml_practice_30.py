# ml_practice_30.py

import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Model
model = LogisticRegression(max_iter=1000)

# K-Fold cross validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print("Scores for 5 folds:", scores)
print("Average accuracy:", scores.mean())
