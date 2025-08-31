import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Example dataset
data = pd.DataFrame({
    "Hours_Studied": [1, 2, 3, 4, 5, 6],
    "Pass": [0, 0, 0, 1, 1, 1]
})

X = data[["Hours_Studied"]]
y = data["Pass"]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

pred = model.predict(X)
print("Predictions:", pred)
print("Accuracy:", accuracy_score(y, pred))
