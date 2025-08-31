import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

# Example dataset
data = pd.DataFrame({
    "Hours_Studied": [1, 2, 3, 4, 5, 6],
    "Pass": [0, 0, 0, 1, 1, 1]
})

X = data[["Hours_Studied"]]
y = data["Pass"]

model = DecisionTreeClassifier()
model.fit(X, y)

pred = model.predict(X)
print("Predictions:", pred)
print("Accuracy:", accuracy_score(y, pred))
print("\nDecision Tree Rules:\n", export_text(model, feature_names=["Hours_Studied"]))
