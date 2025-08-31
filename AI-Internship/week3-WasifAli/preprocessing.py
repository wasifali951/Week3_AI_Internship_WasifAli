# preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Example dataset
data = pd.DataFrame({
    "Name": ["Ali", "Sara", "Ahmed", "Zara", None],
    "Gender": ["Male", "Female", "Male", "Female", "Female"],
    "Age": [20, None, 22, 23, 21],
    "Marks": [85, 90, None, 88, 92]
})

print("Original Data:\n", data)

# Handle missing values
imputer_mean = SimpleImputer(strategy="mean")
data["Age"] = imputer_mean.fit_transform(data[["Age"]])
data["Marks"] = imputer_mean.fit_transform(data[["Marks"]])
data["Name"] = data["Name"].fillna("Unknown")

# Encode categorical
le = LabelEncoder()
data["Gender_Label"] = le.fit_transform(data["Gender"])

# One-Hot Encoding
data = pd.get_dummies(data, columns=["Gender"], drop_first=True)

# Normalize numerical features
scaler = StandardScaler()
data[["Age", "Marks"]] = scaler.fit_transform(data[["Age", "Marks"]])

print("\nPreprocessed Data:\n", data)
