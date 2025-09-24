import os
import yaml
import torch
from src.predictive_maintenance.data import load_data, add_engineered_features, scale, create_feature_cols
from src.predictive_maintenance.datasets import make_dataloader
from src.predictive_maintenance.models import LSTMModel, train_lstm_model
import torch

# -------------------------------
# 1. CONFIG
# -------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "src", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

TRAIN_PATH = config["data"]["train_path"]
TEST_PATH = config["data"]["test_path"]
RUL_PATH = config["data"]["rul_path"]
WINDOW_SIZE = config["training"]["window_size"]
BATCH_SIZE = config["training"]["batch_size"]
EPOCHS = config["training"]["epochs"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1️⃣ Feature selection
feature_cols = create_feature_cols(TRAIN_PATH, TEST_PATH, RUL_PATH)

# 2️⃣ Load & engineer data
train_df, test_df, _ = load_data(TRAIN_PATH, TEST_PATH, RUL_PATH)
train_df = add_engineered_features(train_df)
test_df  = add_engineered_features(test_df)

# -------------------------------
# 3. DATA LOADERS
# -------------------------------
train_loader = make_dataloader(train_df, feature_cols, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE, shuffle=True)
val_loader = make_dataloader(train_df, feature_cols, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------------
# 4. TRAIN MODEL
# -------------------------------
input_dim = len(feature_cols)
model, history = train_lstm_model(train_loader, val_loader, input_dim, epochs=EPOCHS, device=DEVICE)

# -------------------------------
# 5. SAVE BEST MODEL
# -------------------------------
os.makedirs("models/pm", exist_ok=True)
MODEL_PATH = "models/pm/best.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Best model saved at {MODEL_PATH}")
