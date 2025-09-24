import os
import yaml
import torch
from src.predictive_maintenance.data import load_data, add_engineered_features, scale, create_feature_cols
from src.predictive_maintenance.datasets import make_dataloader
from src.predictive_maintenance.train_utils import run_pipeline
import torch

# -------------------------------
# 1. CONFIG
# -------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "src", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


path="data/CMaps"
# Paths for FD001 dataset
train_path = config["data"]["train_path"]
test_path  = config["data"]["test_path"]
rul_path   = config["data"]["rul_path"]
save_path  = "models/"
# Step 1: Create feature columns (includes feature engineering + cleanup)
feature_cols = create_feature_cols(train_path, test_path, rul_path)

# Step 2: Run pipeline end-to-end
model, history, y_pred, y_true = run_pipeline(
    train_path, test_path, rul_path,
    feature_cols=feature_cols,
    epochs=2,
    window_size=30, 
    device="cuda" if torch.cuda.is_available() else "cpu",
)
