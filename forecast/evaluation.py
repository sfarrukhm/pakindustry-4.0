import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings("ignore")

# ===== Metrics =====
def smape(y_true, y_pred):
    return np.mean(
        2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    ) * 100

def nrmse(y_true, y_pred):
    return np.sqrt(root_mean_squared_error(y_true, y_pred)) / np.mean(y_true)

# ===== Load validation data =====
val = pd.read_csv("results/validation_data.csv")  # save this during training
y_val = val["orders"]
preds = val["preds"]

# ===== Evaluate =====
rmse = root_mean_squared_error(y_val, preds)
mae = mean_absolute_error(y_val, preds)
print(f"Validation RMSE: {rmse:.2f}")
print(f"Validation MAE:  {mae:.2f}")
print(f"Validation sMAPE: {smape(y_val, preds):.2f}%")
print(f"Validation NRMSE: {nrmse(y_val, preds):.3f}")
