import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Example dataset
data = pd.DataFrame({
    "Hours_Studied": [2, 3, 4, 5, 6],
    "Marks": [50, 60, 65, 70, 80]
})

X = data[["Hours_Studied"]]
y = data["Marks"]

model = LinearRegression()
model.fit(X, y)

pred = model.predict(X)
print("Predictions:", pred)
print("R² Score:", r2_score(y, pred))
print("MSE:", mean_squared_error(y, pred))
