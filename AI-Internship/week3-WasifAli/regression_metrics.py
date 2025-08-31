from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

X = np.array([[i] for i in range(1, 11)])
y = [0,0,0,0,0,1,1,1,1,1]

# Underfitting (very shallow tree)
underfit = DecisionTreeClassifier(max_depth=1)
underfit.fit(X, y)
print("Underfitting Accuracy:", accuracy_score(y, underfit.predict(X)))

# Overfitting (very deep tree)
overfit = DecisionTreeClassifier(max_depth=None)
overfit.fit(X, y)
print("Overfitting Accuracy:", accuracy_score(y, overfit.predict(X)))
