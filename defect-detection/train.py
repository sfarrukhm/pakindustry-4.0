import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
import seaborn as sns

import yaml
from src.dataset import CastDefectDataset   

# -------------------------------
# 1. CONFIGURATION & SEEDING
# -------------------------------
with open("defect-detection/src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)



# Always resolve relative to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SAVE_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

CSV_TRAIN = os.path.join(DATA_DIR, "train", "_classes.csv")
CSV_VALID = os.path.join(DATA_DIR, "valid", "_classes.csv")
IMG_DIR_TRAIN = os.path.join(DATA_DIR, "train")
IMG_DIR_VALID = os.path.join(DATA_DIR, "valid")
SAVE_PATH = os.path.join(SAVE_DIR, "best_model.pth")


BATCH_SIZE = config['default']['batch_size']
IMG_SIZE = config["default"]["img_size"]
EPOCHS = config["default"]["epochs"]
LR = config["default"]["lr"]
PATIENCE = config["default"]["patience"]

# -------------------------------
# 2. DATA PREPARATION
# -------------------------------
# Load CSVs
df_train = pd.read_csv(CSV_TRAIN)
df_train["label"] = df_train["def_front"]

df_valid = pd.read_csv(CSV_VALID)
df_valid["label"] = df_valid["def_front"]

# Transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = CastDefectDataset(df_train, IMG_DIR_TRAIN, transform)
val_dataset = CastDefectDataset(df_valid, IMG_DIR_VALID, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=2)


# -------------------------------
# 3. MODEL SETUP
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = models.efficientnet_b0(weights=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------------------------
# 4. TRAINING LOOP
# -------------------------------
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(EPOCHS):
    # ---- Training ----
    model.train()
    train_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        images, labels = images.to(device), labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # ---- Validation ----
    model.eval()
    val_loss, preds, targets = 0, [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Valid]"):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds.extend((probs > 0.5).cpu().numpy())
            targets.extend(labels.cpu().numpy())

    # ---- Metrics ----
    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds)
    rec = recall_score(targets, preds)
    f1 = f1_score(targets, preds)
    auc = roc_auc_score(targets, preds)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Val Loss: {val_loss/len(val_loader):.4f} | "
          f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | "
          f"F1: {f1:.4f} | AUC: {auc:.4f}")

    # ---- Early Stopping ----
    avg_val_loss = val_loss / len(val_loader)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"✅ Model improved, saved to {SAVE_PATH}")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"⚠️ No improvement ({patience_counter}/{PATIENCE} patience)")
        if patience_counter >= PATIENCE:
            print("⏹️ Early stopping triggered")
            break

# -------------------------------
# 6. FINAL EVALUATION
# -------------------------------
print("\n📊 Final Model Evaluation")
cm = confusion_matrix(targets, preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["OK", "Defected"],
            yticklabels=["OK", "Defected"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

# Ensure reports/ directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

plot_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"✅ Confusion matrix saved to {plot_path}")

# Show (only works if GUI / notebook is available)
plt.show()
