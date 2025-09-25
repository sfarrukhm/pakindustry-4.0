import pandas as pd
import numpy as np
import joblib

# ===== Load model =====
model = joblib.load("models/best_model.pkl")

# ===== Load test data =====
test = pd.read_csv("data/forecast/test.csv")
train_cal = pd.read_csv("data/forecast/train_calendar.csv")

test["date"] = pd.to_datetime(test["date"])
train_cal["date"] = pd.to_datetime(train_cal["date"])

# Merge calendar
dup_cols = [c for c in train_cal.columns if c in test.columns and c != "date"]
train_cal = train_cal.drop(columns=dup_cols)
test = test.merge(train_cal, on="date", how="left")

# ===== Feature Engineering (same as train) =====
def create_features(df):
    df = df.copy()
    df["weekday"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["warehouse_weekday"] = df["warehouse"] + "_" + df["weekday"].astype(str)
    return df

test = create_features(test)

# NOTE: No lags available in test (future), so only calendar + date features used
features = [c for c in test.columns if c not in ["id","orders","date"]]

# ===== Predict =====
test["orders"] = model.predict(test[features])

# Save predictions
import os
os.makedirs("results", exist_ok=True)
test[["id","orders"]].to_csv("results/predictions.csv", index=False)
print("✅ Predictions saved to results/predictions.csv")
