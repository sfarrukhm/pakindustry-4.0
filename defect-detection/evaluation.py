import os
import time
import torch
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve, auc
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

# Paths
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pth")
CSV_VALID = os.path.join(PROJECT_ROOT, "data", "valid", "_classes.csv")
IMG_DIR_VALID = os.path.join(PROJECT_ROOT, "data", "valid")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Params from config
IMG_SIZE = config["training"]["image_size"]
DEVICE = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")

# Transform
base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize(mean=config["transforms"]["normalize_mean"],
                         std=config["transforms"]["normalize_std"])
])

def preprocess_image(img: Image.Image) -> torch.Tensor:
    """Preprocess a PIL image for inference"""
    image = img.convert("L")                              # grayscale
    image = transforms.ToTensor()(image).repeat(3, 1, 1)  # expand channels
    image = base_transform(image)
    return image.unsqueeze(0).to(DEVICE)

# -------------------------------
# 2. LOAD MODEL
# -------------------------------
arch = config["model"]["architecture"]
if arch == "efficientnet-b0":
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
else:
    raise ValueError(f"Model {arch} not yet supported.")

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

print(f"✅ Model loaded: {arch} from {MODEL_PATH} on {DEVICE}")

# -------------------------------
# 3. EVALUATION
# -------------------------------
df_valid = pd.read_csv(CSV_VALID)
df_valid["label"] = df_valid["def_front"]

y_true, y_pred, y_prob, filenames = [], [], [], []

start_time = time.time()
for _, row in df_valid.iterrows():
    img_path = os.path.join(IMG_DIR_VALID, row["filename"])
    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path)
    tensor = preprocess_image(img)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()

    pred_label = 1 if prob > 0.5 else 0

    y_true.append(row["label"])
    y_pred.append(pred_label)
    y_prob.append(prob)
    filenames.append(row["filename"])

elapsed = time.time() - start_time

# -------------------------------
# 4. METRICS
# -------------------------------
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
rocauc = roc_auc_score(y_true, y_prob)

print("\n📊 Evaluation Results:")
print(f"  Accuracy : {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1-score : {f1:.4f}")
print(f"  ROC-AUC  : {rocauc:.4f}")
print(f"⏱️  Evaluation Time: {elapsed:.2f} sec")

# Save metrics to CSV
metrics_path = os.path.join(RESULTS_DIR, "metrics.csv")
pd.DataFrame([{
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "roc_auc": rocauc,
    "eval_time_sec": elapsed
}]).to_csv(metrics_path, index=False)
print(f"✅ Metrics saved at {metrics_path}")

# -------------------------------
# 5. PLOTS
# -------------------------------
# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=100, bbox_inches="tight")
plt.close()

# PR Curve
precision, recall, _ = precision_recall_curve(y_true, y_prob)
pr_auc = auc(recall, precision)
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, color="green", lw=2, label=f"PR Curve (AUC = {pr_auc:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")
plt.savefig(os.path.join(RESULTS_DIR, "precision_recall_curve.png"), dpi=100, bbox_inches="tight")
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
labels = ["OK", "Defected"]

print("\n🧮 Confusion Matrix (Console View):")
print(f"{'':>12}  Predicted OK | Predicted Defected")
print(f"{'-'*40}")
print(f"Actual OK       {cm[0][0]:>10} | {cm[0][1]:>17}")
print(f"Actual Defected {cm[1][0]:>10} | {cm[1][1]:>17}")

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_eval.png"), dpi=100, bbox_inches="tight")
plt.close()

# -------------------------------
# 6. MISCLASSIFIED ANALYSIS
# -------------------------------
preds_df = pd.DataFrame({
    "filename": filenames,
    "true_label": y_true,
    "pred_label": y_pred,
    "probability": y_prob
})

# Save all predictions
preds_csv = os.path.join(RESULTS_DIR, "all_predictions.csv")
preds_df.to_csv(preds_csv, index=False)
print(f"✅ Predictions saved at {preds_csv}")

# Misclassified only
misclassified = preds_df[preds_df["true_label"] != preds_df["pred_label"]]
if not misclassified.empty:
    misclf_csv = os.path.join(RESULTS_DIR, "misclassified_samples.csv")
    misclassified.to_csv(misclf_csv, index=False)
    print(f"✅ Misclassified details saved at {misclf_csv}")

    # Grid plot
    N = min(10, len(misclassified))
    cols = 5
    rows = (N + cols - 1) // cols
    plt.figure(figsize=(15, 3 * rows))

    for i, row in enumerate(misclassified.head(N).itertuples(), 1):
        img_path = os.path.join(IMG_DIR_VALID, row.filename)
        img = Image.open(img_path)

        plt.subplot(rows, cols, i)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(
            f"T:{'Defected' if row.true_label==1 else 'OK'} | "
            f"P:{'Defected' if row.pred_label==1 else 'OK'}\n"
            f"Conf:{row.probability:.2f}",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "misclassified_samples.png"), dpi=100)
    plt.close()
    print(f"✅ Misclassified samples grid saved at {RESULTS_DIR}/misclassified_samples.png")
else:
    print("🎉 No misclassified samples found (perfect predictions!)")
# -------------------------------
# 7. GENERATE MARKDOWN REPORT
# -------------------------------
report_path = os.path.join(RESULTS_DIR, "evaluation_report.md")

with open(report_path, "w") as f:
    f.write("# 📊 Defect Detection Evaluation Report\n\n")

    f.write("## ✅ Metrics\n")
    f.write(f"- Accuracy: **{acc:.4f}**\n")
    f.write(f"- Precision: **{prec:.4f}**\n")
    f.write(f"- Recall: **{rec:.4f}**\n")
    f.write(f"- F1-score: **{f1:.4f}**\n")
    f.write(f"- ROC-AUC: **{rocauc:.4f}**\n")
    f.write(f"- Evaluation Time: **{elapsed:.2f} sec**\n\n")

    f.write("## 🧮 Confusion Matrix\n")
    f.write(f"```\n")
    f.write(f"{'':>12}  Predicted OK | Predicted Defected\n")
    f.write(f"{'-'*40}\n")
    f.write(f"Actual OK       {cm[0][0]:>10} | {cm[0][1]:>17}\n")
    f.write(f"Actual Defected {cm[1][0]:>10} | {cm[1][1]:>17}\n")
    f.write(f"```\n\n")
    f.write(f"![Confusion Matrix](confusion_matrix_eval.png)\n\n")

    f.write("## 📈 Curves\n")
    f.write(f"- [ROC Curve](roc_curve.png)\n")
    f.write(f"- [Precision-Recall Curve](precision_recall_curve.png)\n\n")

    f.write("## 🔍 Misclassified Samples\n")
    if not misclassified.empty:
        f.write(f"- Total Misclassified: **{len(misclassified)}**\n")
        f.write(f"- [CSV with details](misclassified_samples.csv)\n")
        f.write(f"- Example grid plot:\n\n")
        f.write(f"![Misclassified Samples](misclassified_samples.png)\n\n")
    else:
        f.write("🎉 No misclassified samples found (perfect predictions!)\n\n")

    f.write("## 📂 Data Exports\n")
    f.write("- [Metrics CSV](metrics.csv)\n")
    f.write("- [All Predictions CSV](all_predictions.csv)\n")

print(f"📝 Evaluation report generated at {report_path}")
