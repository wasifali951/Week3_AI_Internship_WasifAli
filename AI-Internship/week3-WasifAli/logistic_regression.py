import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Example dataset
data = pd.DataFrame({
    "Hours_Studied": [1, 2, 3, 4, 5, 6],
    "Pass": [0, 0, 0, 1, 1, 1]  # 0=Fail, 1=Pass
})

X = data[["Hours_Studied"]]
y = data["Pass"]

model = LogisticRegression()
model.fit(X, y)

pred = model.predict(X)
print("Predictions:", pred)
print("Confusion Matrix:\n", confusion_matrix(y, pred))
print("Accuracy:", accuracy_score(y, pred))
