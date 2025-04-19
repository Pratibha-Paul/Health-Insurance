import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
df = pd.read_csv("ins_dataset.csv")

# Convert categorical variables to numeric
df = pd.get_dummies(df, drop_first=True)

# Split dataset
X = df.drop(columns=["charges"])
y = df["charges"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "predictor/insurance_model.pkl")
joblib.dump(X.columns, "predictor/model_columns.pkl")

print("Model training complete")