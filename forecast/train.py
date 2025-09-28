import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import joblib
import os
import random
import json
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


# ========== Helper functions ==========
def smape(y_true, y_pred):
    return np.mean(
        2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    ) * 100

def nrmse(y_true, y_pred):
    return np.sqrt(root_mean_squared_error(y_true, y_pred)) / np.mean(y_true)

def create_features(df):
    df = df.copy()
    df["weekday"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["warehouse_weekday"] = df["warehouse"] + "_" + df["weekday"].astype(str)
    return df

def add_lags(df, lags=[7,14,21,28,35], windows=[7,14,21,28,35]):
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("warehouse")["orders"].shift(lag)
    for w in windows:
        df[f"rmean_{w}"] = (
            df.groupby("warehouse")["orders"].shift(1).rolling(w).mean()
        )
    return df

# ========== Load Data ==========
train = pd.read_csv("data/forecast/train.csv")
train_cal = pd.read_csv("data/forecast/train_calendar.csv")
test = pd.read_csv("data/forecast/test.csv")
test_cal = pd.read_csv("data/forecast/test_calendar.csv")

test_cal["date"] = pd.to_datetime(test_cal["date"])

train["date"] = pd.to_datetime(train["date"])
train_cal["date"] = pd.to_datetime(train_cal["date"])
test["date"] = pd.to_datetime(test["date"])

# Remove duplicate columns from calendar
dup_cols = [c for c in train_cal.columns if c in train.columns and c != "date"]
train_cal = train_cal.drop(columns=dup_cols)
# Merge calendar
train = train.merge(train_cal, on="date", how="left")
test = test.merge(train_cal, on="date", how="left")

# ========== Feature Engineering ==========
train = create_features(train)
test = create_features(test)

full_data = pd.concat([train, test], sort=False)
full_data = add_lags(full_data)

# Categoricals
for col in full_data.select_dtypes(include="object").columns:
    full_data[col] = full_data[col].astype("category")

train = full_data.loc[full_data["orders"].notna()]
test = full_data.loc[full_data["orders"].isna()]
# Save test set
test.to_csv("data/forecast/processed_test.csv", index=False)
# ========== Validation Split ==========
cutoff = train["date"].max() - pd.Timedelta(days=14)
train_set = train[train["date"] <= cutoff]
val_set = train[train["date"] > cutoff]

features = [c for c in test.columns if c not in ["id","orders","date"]]
X_train, y_train = train_set[features], train_set["orders"]
X_val, y_val = val_set[features], val_set["orders"]

# ========== Train LightGBM ==========
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": 42,
    "verbose": -1,
}

model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    num_boost_round=1000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100),
    ],
)

# ========== Evaluate ==========
preds = model.predict(X_val)
rmse = root_mean_squared_error(y_val, preds)
mae = mean_absolute_error(y_val, preds)
print(f"Validation RMSE: {rmse:.2f}")
print(f"Validation MAE:  {mae:.2f}")
print(f"Validation sMAPE: {smape(y_val, preds):.2f}%")
print(f"Validation NRMSE: {nrmse(y_val, preds):.3f}")


# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/best_model.pkl")

# Save validation results
results = {
    "rmse": float(rmse),
    "mae": float(mae),
    "smape": float(smape(y_val, preds)),
    "nrmse": float(nrmse(y_val, preds))
}
os.makedirs("results", exist_ok=True)
with open("results/val_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("✅ Model and metrics saved.")