# Predictive Maintenance: Turbofan Engine RUL Prediction

Deep learning-based remaining useful life (RUL) prediction for jet engines using NASA C-MAPSS dataset

---


## Executive Summary

This project implements an **end-to-end predictive maintenance solution pipeline** for jet engines using deep learning techniques. Built on NASA's Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) turbofan engine degradation dataset, the system predicts Remaining Useful Life (RUL) by analyzing multivariate sensor time series data.

The solution integrates:

* **Data preprocessing & feature engineering**
* **Deep learning model training (LSTM/GRU/CNN-LSTM variants)**
* **Evaluation with domain-specific metrics (MAE, RMSE, MAPE, NASA score)**
* **Inference pipeline for new engine data**
* **Streamlit-based interactive dashboard** for end users

⚠️ **Important Note:** Current model predictions tend to converge to a conservative baseline (~21 cycles) rather than fully capturing nuanced degradation patterns. While numerical metrics appear reasonable, this behavior indicates underfitting. The real achievement here is delivering a **working predictive maintenance pipeline** that can be iteratively improved.

---

## Problem Statement & Business Value

**Challenge:** Unplanned aircraft engine failures can result in:

* Flight cancellations and delays costing airlines millions
* Safety risks to passengers and crew
* Emergency maintenance requiring expensive parts and labor
* Regulatory compliance issues and potential fleet grounding

**Solution Approach:** Predictive maintenance using sensor data analysis to:

* Estimate engine degradation trends
* Enable proactive maintenance scheduling
* Reduce downtime and operational costs
* Provide a proof-of-concept for data-driven aerospace maintenance

---

## Repository Structure

```
predictive-maintenance/
├── data/                        # NASA C-MAPSS dataset
├── models/                      # Trained model checkpoints
├── results/                     # Metrics, plots, and analysis
├── src/
│   └── predictive_maintenance/
│       ├── data.py              # Data loading & preprocessing
│       ├── models.py            # Model architecture
│       ├── utils.py             # Helper functions
│       └── config.yaml          # Configuration
├── train.py                     # Training pipeline
├── inference.py                 # Inference pipeline
├── evaluation.py                # Model evaluation
├── app.py                       # Streamlit dashboard

```

---

## Dataset Description

**NASA C-MAPSS Turbofan Engine Degradation Simulation**

* **Subset Used:** FD001 (single operating condition, single fault mode)
* **Training Engines:** 100 engines (full run-to-failure)
* **Test Engines:** 100 engines (truncated runs)
* **Measurements:** 21 sensors + 3 operating conditions
* **Characteristics:** Engines start healthy and degrade over time with noise and variation
*  **Source**: [NASA Turbofan Jet Engine Data Set - Run to Failure Degradation Simulation - Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

---

## Technical Architecture

### Data Pipeline

* Feature engineering (derived ratios, rolling stats)
* Normalization via standard scaling
* Sequence generation using sliding windows (30 cycles)

### Model Architectures Tested

* LSTM + GRU hybrid
* CNN-LSTM combination
* Dropout and layer normalization for regularization

### Evaluation Metrics

* **MAE (Mean Absolute Error)** – average error in cycles
* **RMSE (Root Mean Squared Error)** – penalizes larger errors
* **MAPE (Mean Absolute Percentage Error)** – relative error (%)
* **NASA Scoring Function** – domain-specific penalty for over estimation
---

## Performance Metrics & Results

### Validation & Test Metrics

* **Validation Set:** MAE ~11 cycles, RMSE ~16 cycles, MAPE ~20%
* **Split Test Set:** MAE ~12 cycles, RMSE ~18 cycles, MAPE ~21%
* **NASA Official Test Set:** MAE ~13 cycles, RMSE ~19 cycles, MAPE ~22%, NASA Score ~680

### Observed Prediction Behavior

* Predictions tend to converge to ~21 cycles across engines
* Model underfits the complex degradation patterns
* Conservative outputs may minimize extreme penalties (NASA score) but lack practical resolution

### Interpretation

While the metrics suggest “reasonable” performance, the flat predictions indicate the current model is effectively a **conservative baseline predictor**. This limits its practical utility, but the pipeline remains valid and can be improved with:

* Better architecture (attention, transformers)
* Hyperparameter tuning
* Training across multiple datasets 

---

## Model Limitations

* **Flat predictions (~21 cycles)** highlight underfitting
* Trained only on FD001 subset – generalization unknown
* Fixed 30-cycle windows may not capture long-term dependencies
* Predictions lack uncertainty quantification

---

## Recommended Improvements

* **Short-term:** Try attention mechanisms, hyperparameter tuning, sequence length optimization
* **Medium-term:** Extend training to FD002–FD004, add ensemble models
* **Long-term:** Deploy transformers or Bayesian models for robust predictions with uncertainty bounds

---

## Reproducibility & Code Guidelines

Follow these steps to reproduce results and run the system:


### 1. Clone Repository

```bash
git clone https://github.com/sfarrukhm/pakindustry-4.0.git
cd pakindustry-4.0
```

### 2. Setup Environment

```bash
python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
.venv\Scripts\activate      # (Windows)
pip install -r requirements.txt
cd predictive-maintenance
```

### 3. Train Model

```bash
python train.py
```

This will generate checkpoints inside `models/`.

### 4. Run inference

```bash
python inference.py --input data/CMaps/test_FD001.txt
```



Upload your test file and interact with predictions visually.

### 5. Evaluate results

```bash
python evaluation.py
```
### 6. Launch the Streamlit dashboard

```bash
streamlit run app.py
```
➡️ [Watch the demo](https://youtu.be/0YBDf9Avagc)

---

## Conclusion

This project delivers a **complete predictive maintenance pipeline**: from data preprocessing and model training to inference and interactive deployment. While the current model underfits (flat predictions), the framework is in place for future improvements.
