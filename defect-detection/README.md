# 🏭 Defect Detection Module - PakIndustry 4.0 AI Suite

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Project Overview

This module is part of the **PakIndustry 4.0 AI Suite** developed for the **Uraan AI Techathon 1.0 - Manufacturing Industry Challenge**. It addresses one of the critical problems facing Pakistani manufacturing: **high wastage due to production defects**.

**What it does:**

- Detects defects in cast parts using deep learning model (EfficientNet-B0)
- Classifies images as **OK** or **Defected** with confidence scores
- Provides comprehensive evaluation metrics and visualizations
- Offers both single-image and batch inference capabilities

**Why it matters:**

- Export-oriented manufacturing in Pakistan loses millions annually due to production defects
- Enhances quality control for international competitiveness
- Designed for mid-sized, locally managed industrial units

**Complete Suite Context:**
This is 1 of 3 modules in the PakIndustry 4.0 AI Suite:

1. **🎯 Defect Detection** (this module) - Computer vision for quality control
2. **🔧 Predictive Maintenance** - IoT-based equipment failure prediction
3. **📊 Supply Chain Forecasting** - AI-driven inventory management

---

## 📁 Repository Structure

```
defect-detection/
│
├── 📁 data/                          # Training and validation datasets
├── 📁 models/                        # Trained model checkpoints
├── 📁 results/                       # Evaluation outputs and visualizations
├── 📁 notebooks/                     # Jupyter notebooks for exploration
├── 📁 src/                           # Source code modules
├── 📄 train.py                       # Main training script
├── 📄 inference.py                   # Inference script for predictions
├── 📄 evaluation.py                  # Model evaluation and metrics
├── 📄 app.py                         # Streamlit web application
├── 📄 requirements.txt               # Python dependencies
└── 📄 README.md                      # This documentation
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12 or higher
- CUDA-compatible GPU (recommended for training)
- 4GB+ RAM

### Installation

1. **Clone the repository**
   
   ```bash
   git clone https://github.com/sfarrukhm/pakindustry-4.0.git
   ```

2. **Create virtual environment** (recommended)
   
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   
   ```bash
   pip install -r requirements.txt
   cd defect-detection
   ```

4. **Verify installation**
   
   ```bash
   python inference.py --help
   ```

---

📊 Dataset Information

### Dataset Details

- **Source**: `Roboflow Universe - Open source casting defects dataset`
- **Dataset ID**: cast-defect-w5mh1
- **Classes**: 2 (OK, Defective)
- **Total Images**: 7,284 images
- **Task Type**: Classification
- **License**: Open source
- **Created**: October 2023

### Train Data Distribution

Total Images in train set: 5101

| Class     | Count | Percentage |
| --------- | ----- | ---------- |
| OK        | 2916  | 57.17%     |
| Defective | 2185  | 42.83%     |

### Dataset Characteristics

- **Domain**: Industrial casting defect detection
- **Quality**: Production-grade industrial images
- **Balance**: Slightly imbalanced between OK and defective samples

### Data Augmentation

- Random horizontal/vertical flips
- Random rotation (±15°)
- Random brightness/contrast adjustment
- Gaussian noise injection
- All applied during training only

### Usage & Access

- **Platform**: Roboflow Universe
- **URL**: https://universe.roboflow.com/casting-defects/cast-defect-w5mh1

### Compliance

- ✅ Open source license
- ✅ Publicly available dataset

## 🎯 Training

### Configuration

Training parameters are defined in `src/config.yaml`:

```yaml
system:
  seed: 42
  device: "cuda"         
  num_workers: 2

model:
  architecture: "efficientnet_b0"
  num_classes: 2
  pretrained: true

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 20
  patience: 3
  image_size: 300

transforms:
  resize: [300, 300]
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

paths:
  data_dir: "./data"
  train_csv: "./data/train/_classes.csv"
  valid_csv: "./data/valid/_classes.csv"
  train_images: "./data/train"
  valid_images: "./data/valid"
  models_dir: "./models"
  results_dir: "./results"
  best_model: "./models/best_model.pth"
```

### Run Training

```bash
# Start training with default config
python train.py
```

### Training Results

```
📌 Final Metrics:
Accuracy : 0.9972
Precision: 1.0000
Recall   : 0.9953
F1-score : 0.9977
AUC      : 0.9977
```

---

## 🔍 Inference

### Single Image Prediction

```bash
# Predict single image
python inference.py --image data/valid/sample.jpg --confidence
```

**Example Output:**

```
Image: data/valid/sample.jpg
Prediction: Defected
Confidence: 1.00 (100.0%)
Processing time: 0.10 seconds
```

### Batch Prediction

```bash

# Process with visualization
python inference.py --folder data/valid/ --output results.csv --visualize## Web Application
```

Access the web interface at `http://localhost:8501` for interactive defect detection.

---

## 📈 Evaluation & Metrics

### Run Evaluation

```bash
# Comprehensive evaluation on test set
python evaluation.py --model models/best_model.pth --data data/test/

# Generate all visualizations
python evaluation.py --model models/best_model.pth --plots --misclassified
```

### Performance Metrics

#### **🎯 Competition Requirements: ✅ EXCEEDED**

| Metric    | **Target** | **Achieved** |
| --------- | ---------- | ------------ |
| Accuracy  | ≥ 85%      | **99.6%**    |
| Precision | -          | **100%**     |
| Recall    | -          | **99.4%**    |
| F1-Score  | -          | **99.7%**    |
| ROC-AUC   | -          | **100%**     |

#### 

## 📊 Results & Visualizations

### 1. Confusion Matrix

![Confusion Matrix](results/confusion_matrix_eval.png)

*Clear separation between classes with minimal false positives/negatives*

### 2. ROC Curve

![](D:\MYWOrk\pakindustry-4.0\defect-detection\results\roc_curve%20-%20Copy.png)

### 3. Precision-Recall Curve

![](D:\MYWOrk\pakindustry-4.0\defect-detection\results\precision_recall_curve%20-%20Copy.png)

*Balanced precision-recall trade-off across all thresholds*

### 4. Misclassified Samples Analysis

![Misclassified Grid](D:\MYWOrk\pakindustry-4.0\defect-detection\results\misclassified_samples.png)

*Analysis of edge cases and challenging samples for continuous improvement*

---

## 🏆 Competition Advantages

### **Technical Excellence**

- **Model Performance**: 99.6% accuracy
- **Robust Architecture**: EfficientNet-B0 optimized for manufacturing defects

### **Pakistan Manufacturing Context**

- **Resource Efficient**: Designed for mid-sized manufacturers with limited IT resources
- **Cost Effective**: Uses readily available hardware configurations

### **Production Readiness**

- **Scalable Architecture**: Easy integration into existing quality control workflows
- **Web Interface**: User-friendly for non-technical factory operators
- **Comprehensive Logging**: Full audit trail for quality compliance

---

## 🔧 Technical Specifications

### Model Architecture

- **Base Model**: EfficientNet-B0 (pre-trained on ImageNet)
- **Custom Head**: Global Average Pooling → Dropout(0.3) → Dense(2) → Softmax
- **Parameters**: ~5.3M (lightweight for deployment)
- **Input Size**: 224×224×3 RGB images
- **Output**: Binary classification with confidence scores

# 

### Data Augmentation

- **Flip:** Horizontal, Vertical

- **90° Rotate:** Clockwise, Counter-Clockwise

- **Rotation**: Between -15° and +15°

- **Shear**: ±15° Horizontal, ±15° Vertical

---

## 🚨 Limitations & Future Work

### Current Limitations

- **Lighting Conditions**: Performance may vary under extreme lighting
- **New Defect Types**: Requires retraining for previously unseen defect patterns
- **Material Specific**: Optimized for cast parts; may need adjustment for other materials
- **Inference Time**: Current inference time per image is 0.27s, which is comparatively large. 

### Planned Enhancements

1. **Multi-class Detection**: Extend to classify specific defect types
2. **Real-time Processing**: Optimize for continuous production line integration
3. **Edge Deployment**: TensorRT/ONNX optimization for industrial edge devices
4. **Active Learning**: Implement feedback loop for continuous model improvement

---

## 📱 Web Application Features

```bash
# Launch Streamlit app
streamlit run app.py
```

### Streamlit Interface

- **Upload & Predict**: Drag-and-drop image upload with instant results
- **Batch Processing**: Upload multiple images for bulk analysis
- **Confidence Visualization**: Interactive confidence score displays
- **Historical Analysis**: Track defect patterns over time
- **Export Reports**: Download results in CSV/PDF format

### Screenshot

![](C:\Users\HP\AppData\Roaming\marktext\images\2025-09-23-16-41-42-image.png)
