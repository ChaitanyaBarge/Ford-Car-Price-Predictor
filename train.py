import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# 1. Load dataset (adjust filename if needed)
df = pd.read_csv("ford.csv.csv")

# Clean dataset column names if necessary
df.columns = df.columns.str.strip()

# 2. Separate target (price) and features
X = df.drop(columns=["price"])
y = df["price"]

# 3. One-Hot Encoding for categorical features
X = pd.get_dummies(X, columns=["model", "transmission", "fuelType"])

# Save exact feature column names
columns = list(X.columns)
joblib.dump(columns, "columns.pkl")

# 4. Scale numerical features
num_cols = ["year", "mileage", "tax", "mpg", "engineSize"]
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
joblib.dump(scaler, "scaler.pkl")

# 5. Train and save Linear Regression Model
model = LinearRegression()
model.fit(X, y)
joblib.dump(model, "LR_ford_car.pkl")

print("Successfully generated LR_ford_car.pkl, scaler.pkl, and columns.pkl!")