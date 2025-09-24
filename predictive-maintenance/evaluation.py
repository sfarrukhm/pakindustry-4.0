import os
import yaml
import torch
import numpy as np
from src.metrics import print_score, plot_scatter, plot_histogram
from src.dataset import make_test_windows
from src.model import LSTMModel
from src.data_utils import load_data, add_engineered_features, scale

# -------------------------------
# CONFIG
# -------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "src", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

FEATURE_COLS = config["model"]["feature_cols"]
WINDOW_SIZE = config["training"]["window_size"]
DEVICE = config["system"]["device"]
MODEL_PATH = "models/pm/best.pth"

# -------------------------------
# LOAD DATA
# -------------------------------
train_df, test_df, test_rul = load_data(config["data"]["train_path"], config["data"]["test_path"], config["data"]["rul_path"])
train_df = add_engineered_features(train_df)
test_df = add_engineered_features(test_df)
train_df, test_df = scale(train_df, test_df, FEATURE_COLS)

# -------------------------------
# LOAD MODEL
# -------------------------------
input_dim = len(FEATURE_COLS)
model = LSTMModel(input_dim).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------------------------------
# EVALUATION
# -------------------------------
X_test, engine_numbers = make_test_windows(test_df, FEATURE_COLS, window_size=WINDOW_SIZE)
X_test = X_test.to(DEVICE)

with torch.no_grad():
    y_pred = model(X_test).squeeze().cpu().numpy()
y_true = test_rul[:len(engine_numbers)]

# Print metrics
print_score(y_true, y_pred)

# Save plots with _eval suffix
plot_scatter(y_true, y_pred, save_path="results/scatter_eval.png")
plot_histogram(y_true, y_pred, save_path="results/hist_eval.png")
