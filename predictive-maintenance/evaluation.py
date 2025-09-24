import os
import yaml
import torch
import numpy as np
from src.predictive_maintenance.metrics import print_score, plot_scatter, plot_histogram
from src.predictive_maintenance.datasets import make_test_windows
from src.predictive_maintenance.models import LSTM_RUL
from src.predictive_maintenance.data import (
    load_data,
    add_engineered_features,
    scale,
    create_feature_cols,   # ✅ consistent with training
)

# -------------------------------
# CONFIG
# -------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "src", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

WINDOW_SIZE = config["data"]["window_size"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/best.pt"

# -------------------------------
# FEATURE COLUMNS
# -------------------------------
FEATURE_COLS = create_feature_cols(
    config["data"]["train_path"],
    config["data"]["test_path"],
    config["data"]["rul_path"],
    max_rul=config["training"].get("max_rul", 125)
)

# -------------------------------
# LOAD DATA
# -------------------------------
train_df, test_df, test_rul = load_data(
    config["data"]["train_path"],
    config["data"]["test_path"],
    config["data"]["rul_path"],
    max_rul=config["training"].get("max_rul", 125)
)
train_df = add_engineered_features(train_df)
test_df = add_engineered_features(test_df)
train_df, test_df = scale(train_df, test_df, FEATURE_COLS)

# -------------------------------
# LOAD MODEL
# -------------------------------
input_dim = len(FEATURE_COLS)
model = LSTM_RUL(input_dim).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------------------------------
# EVALUATION
# -------------------------------
X_test, engine_numbers = make_test_windows(test_df, FEATURE_COLS, window_size=WINDOW_SIZE)
X_test = X_test.to(DEVICE)

with torch.no_grad():
    y_pred = model(X_test).squeeze().cpu().numpy()

y_true = np.array(test_rul[:len(engine_numbers)])

# Save metrics + plots with _eval suffix
print_score(y_true, y_pred, prefix="eval")
plot_scatter(y_true, y_pred, prefix="eval")
plot_histogram(y_true, y_pred, prefix="eval")
