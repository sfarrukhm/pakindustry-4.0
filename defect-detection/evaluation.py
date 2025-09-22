import os
import torch
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import torchvision.transforms as transforms
import torchvision.models as models

# -------------------------------
# 1. CONFIGURATION
# -------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "src", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

IMG_SIZE = config["default"]["img_size"]
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pth")
CSV_VALID = os.path.join(PROJECT_ROOT, "data", "valid", "_classes.csv")
IMG_DIR_VALID = os.path.join(PROJECT_ROOT, "data", "valid")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------
# 2. PREPROCESSING
# -------------------------------
base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(img: Image.Image, device) -> torch.Tensor:
    image = img.convert("L")
    image = transforms.ToTensor()(image).repeat(3, 1, 1)
    image = base_transform(image)
    return image.unsqueeze(0).to(device)

# -------------------------------
# 3. LOAD MODEL
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

print(f"✅ Model loaded from {MODEL_PATH} on {device}")

# -------------------------------
# 4. EVALUATION
# -------------------------------
df_valid = pd.read_csv(CSV_VALID)
df_valid["label"] = df_valid["def_front"]

y_true, y_pred, y_prob = [], [], []

for _, row in df_valid.iterrows():
    img_path = os.path.join(IMG_DIR_VALID, row["filename"])
    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path)
    tensor = preprocess_image(img, device)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()

    pred_label = 1 if prob > 0.5 else 0

    y_true.append(row["label"])
    y_pred.append(pred_label)
    y_prob.append(prob)

# --- Metrics ---
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)

print("\n📊 Evaluation Results:")
print(f"  Accuracy : {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1-score : {f1:.4f}")
print(f"  ROC-AUC  : {auc:.4f}")

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["OK", "Defected"],
            yticklabels=["OK", "Defected"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

cm_path = os.path.join(RESULTS_DIR, "confusion_matrix_eval.png")
plt.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Confusion matrix saved at {cm_path}")
