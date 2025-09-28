# 🏭 Defect Detection Module – PakIndustry 4.0 AI Suite

Computer vision–based **casting defect detection** for manufacturing quality control.

---

## Executive Summary

This module is part of the **PakIndustry 4.0 AI Suite** developed for the **Uraan AI Techathon 1.0 – Manufacturing Industry Challenge**. It addresses a critical pain point in Pakistani manufacturing: **high wastage due to production defects in cast parts**.

**Core Capabilities:**

* Binary classification (**OK** vs **Defected**) using EfficientNet-B0
* Confidence scoring for predictions
* Batch and single-image inference support
* Comprehensive evaluation with visualizations
* Interactive Streamlit dashboard for non-technical users

**Context in Full Suite:**

1. 🎯 **Defect Detection** – Vision-based QC (this module)
2. 🔧 **Predictive Maintenance** – IoT-driven failure prediction
3. 📊 **Supply Chain Forecasting** – AI-powered inventory planning

---

## Problem Statement & Business Value

**Challenge:**

* Export-oriented manufacturing in Pakistan loses millions annually from production defects
* Lack of affordable, automated defect detection tools

**Solution Impact:**

* Automated defect screening for cast parts
* Reduces human error in QC processes
* Scalable for mid-sized manufacturers with limited IT resources
* Boosts competitiveness in global markets

---

## Repository Structure

```
defect-detection/
├── data/              # Training and validation datasets
├── models/            # Saved checkpoints
├── results/           # Evaluation metrics and visualizations
├── notebooks/         # Jupyter notebooks for exploration
├── src/               # Core source code
├── train.py           # Training pipeline
├── inference.py       # Prediction script
├── evaluation.py      # Evaluation metrics & plots
├── app.py             # Streamlit dashboard
├── requirements.txt   # Dependencies
└── README.md
```

---

## Dataset

**Source:** [Roboflow Universe – Cast Defect Dataset](https://universe.roboflow.com/casting-defects/cast-defect-w5mh1)
**Total Images:** 7,284 (train/valid/test split)
**Classes:** 2 (OK, Defected)

| Class     | Count | Percentage |
| --------- | ----- | ---------- |
| OK        | 2916  | 57.2%      |
| Defective | 2185  | 42.8%      |

**Data Characteristics:**

* Domain: Industrial cast parts
* Quality: Production-grade images
* Balance: Slightly imbalanced dataset

**Data Augmentation (training only):**

* Random flips & rotations (±15°)
* Brightness/contrast adjustment
* Gaussian noise injection

---

## Model & Training

**Architecture:** EfficientNet-B0 (ImageNet pretrained)

* Global Average Pooling → Dropout(0.3) → Dense(2) → Softmax
* Parameters: ~5.3M
* Input size: 224×224×3 RGB

**Training Configuration:** (from `src/config.yaml`)

* Batch size: 32
* Learning rate: 0.001
* Epochs: 20 (early stopping patience = 3)
* Image size: 300×300
* Optimized with Adam

**Final Metrics (Validation):**

* Accuracy: **99.7%**
* Precision: **100%**
* Recall: **99.5%**
* F1-score: **99.7%**
* ROC-AUC: **100%**

---

## Inference

### Single Image Prediction

```bash
python inference.py --image data/valid/sample.jpg
```

Output:

```
Prediction: Defected  
Confidence: 100%  
Processing time: 0.10s
```

### Batch Inference

```bash
python inference.py --folder data/valid/ --output results.csv
```

---

## Evaluation

Run:

```bash
python evaluation.py --model models/best_model.pth --data data/test/
```

Generates:

* Confusion matrix
* ROC curve
* Precision-recall curve
* Misclassified samples analysis

**Achieved vs Target (Techathon):**

| Metric    | Target | Achieved  |
| --------- | ------ | --------- |
| Accuracy  | ≥85%   | **99.6%** |
| Precision | -      | **100%**  |
| Recall    | -      | **99.4%** |
| F1-Score  | -      | **99.7%** |
| ROC-AUC   | -      | **100%**  |

---

## Web Application (`app.py`)

Interactive **Streamlit dashboard** for defect detection.

```bash
streamlit run app.py
```

**Features:**

* Upload & predict (single or multiple images)
* Confidence visualization
* Misclassified image inspection
* Downloadable results (CSV/PDF)
* Simple interface for factory floor operators

---

## Limitations

* Sensitive to extreme lighting variations
* Needs retraining for new defect types/materials
* Optimized for cast parts (not yet generalized)
* Inference ~0.27s per image (not real-time for high-speed lines)

---

## Future Work

1. Extend to **multi-class defect detection**
2. Optimize for **real-time processing** (TensorRT/ONNX)
3. Deployable to **edge devices**
4. Integrate **active learning** for continuous improvement

---

## Reproducibility Guidelines

### 1. Clone Repository

```bash
git clone https://github.com/sfarrukhm/pakindustry-4.0.git
cd defect-detection
```

### 2. Setup Environment

```bash
python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
.venv\Scripts\activate      # (Windows)
pip install -r requirements.txt
```

### 3. Train Model

```bash
python train.py
```

### 4. Run Inference

```bash
python inference.py --image data/valid/sample.jpg
```

### 5. Evaluate

```bash
python evaluation.py --model models/best_model.pth --data data/test/
```

### 6. Launch Dashboard

```bash
streamlit run app.py
```

---

## Conclusion

This module delivers a **production-ready computer vision solution** for detecting defects in cast parts. With **99.6% accuracy**, a lightweight EfficientNet backbone, and a user-friendly dashboard, it provides manufacturers with a scalable and cost-effective tool to reduce waste and improve quality control.

