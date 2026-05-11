import pandas as pd
import numpy as np
import joblib
import os
from lightgbm import LGBMRegressor

# create models folder
os.makedirs("models", exist_ok=True)

# load dataset
df = pd.read_csv("House Price.csv")

# feature engineering based on your project
df["LOG_SQFT"] = np.log1p(df["SQUARE_FT"])
df["PRICE_LOG"] = np.log1p(df["TARGET(PRICE_IN_LACS)"])

# one hot encode posted_by
posted = pd.get_dummies(df["POSTED_BY"], prefix="POSTED_BY")

df = pd.concat([df, posted], axis=1)

# ensure required columns exist
if "POSTED_BY_Dealer" not in df.columns:
    df["POSTED_BY_Dealer"] = 0

if "POSTED_BY_Owner" not in df.columns:
    df["POSTED_BY_Owner"] = 0

# final feature set from your project
features = [
    "UNDER_CONSTRUCTION",
    "RERA",
    "BHK_NO.",
    "RESALE",
    "LATITUDE",
    "LONGITUDE",
    "LOG_SQFT",
    "POSTED_BY_Dealer",
    "POSTED_BY_Owner"
]

X = df[features]
y = df["PRICE_LOG"]

# final model
model = LGBMRegressor(
    random_state=42,
    n_estimators=300,
    learning_rate=0.05
)

model.fit(X, y)

joblib.dump(model, "models/lightgbm_model.pkl")
joblib.dump(features, "models/model_features.pkl")

print("Model rebuilt and saved successfully.")
