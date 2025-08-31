import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Example dataset
data = pd.DataFrame({
    "Hours_Studied": [1,2,3,4,5,6,7,8,9,10],
    "Sleep_Hours": [7,6,8,5,6,7,8,6,7,8],
    "Marks": [50,55,60,65,70,75,80,82,85,90]
})

X = data[["Hours_Studied", "Sleep_Hours"]]
y = data["Marks"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

# Decision Tree Regression
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)

print("Linear Regression R²:", r2_score(y_test, pred_lr))
print("Decision Tree R²:", r2_score(y_test, pred_dt))

print("Linear Regression MSE:", mean_squared_error(y_test, pred_lr))
print("Decision Tree MSE:", mean_squared_error(y_test, pred_dt))
# student_performance_model.py