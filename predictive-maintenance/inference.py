import os
import yaml
import torch
from src.predictive_maintenance.datasets import make_test_windows
from src.predictive_maintenance.models import LSTM_RUL
from src.predictive_maintenance.data import load_data, add_engineered_features, scale

# -------------------------------
# CONFIG
# -------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "src", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

TEST_PATH = config["data"]["test_path"]
RUL_PATH = config["data"]["rul_path"]
FEATURE_COLS = config["model"]["feature_cols"]  # should be set after running create_feature_cols
WINDOW_SIZE = config["training"]["window_size"]
DEVICE = config["system"]["device"]
MODEL_PATH = "models/pm/best.pth"

# -------------------------------
# LOAD DATA
# -------------------------------
train_df, test_df, test_rul = load_data(config["data"]["train_path"], TEST_PATH, RUL_PATH)
test_df = add_engineered_features(test_df)
train_df = add_engineered_features(train_df)
train_df, test_df = scale(train_df, test_df, FEATURE_COLS)

# -------------------------------
# LOAD MODEL
# -------------------------------
input_dim = len(FEATURE_COLS)
model = LSTM_RUL(input_dim).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------------------------------
# MAKE TEST WINDOWS
# -------------------------------
X_test, engine_numbers = make_test_windows(test_df, FEATURE_COLS, window_size=WINDOW_SIZE)
X_test = X_test.to(DEVICE)

with torch.no_grad():
    y_pred = model(X_test).squeeze().cpu().numpy()

y_true = test_rul[:len(engine_numbers)]
print("✅ Inference completed. Predictions ready.")

# Optionally: save to CSV
import pandas as pd
df_preds = pd.DataFrame({"engine_number": engine_numbers, "true_RUL": y_true, "pred_RUL": y_pred})
df_preds.to_csv("results/pm_inference.csv", index=False)
print("✅ Predictions saved at results/pm_inference.csv")
