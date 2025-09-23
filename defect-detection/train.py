import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from tqdm import tqdm
from PIL import Image

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,roc_curve, precision_recall_curve
)

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from src.dataset import CastDefectDataset


# -------------------------------
# 1. CONFIGURATION & SEEDING
# -------------------------------
with open("/src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

SEED = config["system"]["seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SAVE_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CSV_TRAIN = os.path.join(DATA_DIR, "train", "_classes.csv")
CSV_VALID = os.path.join(DATA_DIR, "valid", "_classes.csv")
IMG_DIR_TRAIN = os.path.join(DATA_DIR, "train")
IMG_DIR_VALID = os.path.join(DATA_DIR, "valid")
SAVE_PATH = os.path.join(SAVE_DIR, "best_model.pth")

# Hyperparameters
BATCH_SIZE = config["training"]["batch_size"]
IMG_SIZE = config["training"]["image_size"]
EPOCHS = config["training"]["epochs"]
LR = config["training"]["learning_rate"]
PATIENCE = config["training"]["patience"]


# -------------------------------
# 2. DATA PREPARATION
# -------------------------------
df_train = pd.read_csv(CSV_TRAIN)
df_train["label"] = df_train["def_front"]

df_valid = pd.read_csv(CSV_VALID)
df_valid["label"] = df_valid["def_front"]

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize(mean=config["transforms"]["mean"],
                         std=config["transforms"]["std"])
])

train_dataset = CastDefectDataset(df_train, IMG_DIR_TRAIN, transform)
val_dataset = CastDefectDataset(df_valid, IMG_DIR_VALID, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# -------------------------------
# 3. MODEL SETUP
# -------------------------------
if config["model"]["architecture"] == "efficientnet-b0":
    model = models.efficientnet_b0(weights="IMAGENET1K_V1" if config["model"]["pretrained"] else None)
else:
    raise ValueError(f"Model {config['model']['architecture']} not supported yet.")

# Replace classifier
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, config["model"]["num_classes"] - 1)  # binary → 1 output
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# -------------------------------
# 4. TRAINING LOOP
# -------------------------------
best_val_loss = float("inf")
patience_counter = 0

history = {
    "train_loss": [], "val_loss": [],
    "accuracy": [], "precision": [],
    "recall": [], "f1": [], "auc": []
}

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

            probs = torch.sigmoid(outputs).cpu().numpy()
            preds.extend((probs > 0.5).astype(int))
            targets.extend(labels.cpu().numpy())

    # ---- Metrics ----
    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, zero_division=0)
    rec = recall_score(targets, preds, zero_division=0)
    f1 = f1_score(targets, preds, zero_division=0)
    auc = roc_auc_score(targets, preds)

    history["train_loss"].append(train_loss / len(train_loader))
    history["val_loss"].append(val_loss / len(val_loader))
    history["accuracy"].append(acc)
    history["precision"].append(prec)
    history["recall"].append(rec)
    history["f1"].append(f1)
    history["auc"].append(auc)

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
# 5. FINAL EVALUATION
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

plot_path = os.path.join(RESULTS_DIR, "confusion_matrix_train.png")
plt.savefig(plot_path, dpi=100, bbox_inches="tight")
print(f"✅ Confusion matrix saved to {plot_path}")
# -------------------------------
# 6. ROC Curve
# -------------------------------
fpr, tpr, _ = roc_curve(targets, preds)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
roc_path = os.path.join(RESULTS_DIR, "roc_curve_train.png")
plt.savefig(roc_path, dpi=100, bbox_inches="tight")
print(f"✅ ROC curve saved to {roc_path}")
plt.close()

# -------------------------------
# 7. Precision-Recall Curve
# -------------------------------
precisions, recalls, _ = precision_recall_curve(targets, preds)
plt.figure(figsize=(6, 5))
plt.plot(recalls, precisions)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
pr_path = os.path.join(RESULTS_DIR, "precision_recall_curve_train.png")
plt.savefig(pr_path, dpi=100, bbox_inches="tight")
print(f"✅ Precision-Recall curve saved to {pr_path}")
plt.close()
# Final Summary
print("\n📌 Final Metrics:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")

plt.show()
