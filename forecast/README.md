# Rohlik Orders Forecasting

Advanced **supply chain demand forecasting** using machine learning and time series analysis.

---

## Executive Summary

This project implements an **end-to-end forecasting pipeline** for daily orders across 7 European warehouses (Prague, Brno, Munich, Frankfurt, Budapest) using **3+ years of historical data** (Dec 2020 – Mar 2024).

The solution integrates:

* **Data preprocessing & feature engineering** (lags, rolling statistics, calendar events)
* **LightGBM-based forecasting model**
* **Evaluation with scale-independent metrics (sMAPE, NRMSE)**
* **Inference pipeline for new test sets**
* **Streamlit interactive dashboard** (`app.py`) for uploading data and viewing predictions

⚠️ **Important Note:** While performance is exceptionally strong (sMAPE ~1.97%), the model depends heavily on historical patterns and may underperform if demand behaviors shift unexpectedly. The main achievement is a **working, reproducible forecasting pipeline** that generalizes well within observed conditions.

---

## Problem Statement & Business Value

**Challenge:** Accurate warehouse-level demand forecasting is critical for:

* Inventory management (reducing overstocking and stockouts)
* Workforce planning for daily operations
* Supply chain cost optimization across multiple regions
* Meeting customer demand consistently

**Solution Approach:** Machine learning–driven forecasting to:

* Capture seasonality and operational event effects
* Forecast daily order volumes per warehouse
* Reduce planning errors below the 10% competition threshold
* Enable robust multi-country supply chain operations

---

## Repository Structure

```
forecast/
├── data/forecast/        # Training and test datasets
├── models/               # Trained model checkpoints
├── results/              # Metrics and predictions
├── train.py              # Model training pipeline
├── inference.py          # Batch inference script
├── evaluation.py         # Model evaluation
├── app.py                # Streamlit dashboard for predictions
```

---

## Dataset Description

**Primary Files:**

* `train.csv` – Historical daily orders per warehouse (Dec 2020 – Mar 2024)
* `train_calendar.csv` – Calendar event flags (holidays, closures, shutdowns, school holidays)
* `test.csv` – Test set warehouse-date pairs for forecasting

**Details:**

* **Target Variable:** Daily total orders per warehouse
* **Warehouses:** Prague_1, Brno_1, Prague_2, Prague_3, Munich_1, Frankfurt_1, Budapest_1
* **Features:** Lagged orders, rolling means, weekday/month/holiday effects, warehouse encodings
* **Calendar Integration:** Seasonal and operational events merged on date
* **Source:** [Rohlik Orders Forecasting Challenge - Kaggle](https://www.kaggle.com/competitions/rohlik-orders-forecasting-challenge/data)

---

## Technical Architecture

### Data Pipeline

* Calendar event merging
* Feature engineering (temporal, lags, rolling stats)
* Encoding of categorical warehouses

### Model

* **Algorithm:** LightGBM regressor
* **Validation:** Last 14 days as holdout
* **Reproducibility:** Fixed seed across all runs


## Performance Metrics & Results

**Validation Performance:**

* **sMAPE:** 1.97%
* **NRMSE:** 0.032
* * **RMSE:** 208.40
* **MAE:** 126.87

**Performance Context:**
Daily orders range **5,000–8,500 per warehouse**. The achieved error rates correspond to **2–3% relative error**, significantly outperforming the hackathon threshold of **≤10%**.

---

## Model Limitations

* Relies on continuation of historical demand patterns
* May underperform on unseen events (new holidays, strikes, market shifts)
* Holidays and anomalies not present in training data are harder to model
* Training-only features (e.g., weather, user activity) excluded from inference

---

## Recommended Improvements

* Cyclical encoding for temporal features (sine/cosine transforms)
* Rolling feature expansion (standard deviation, min, max)
* Days-to-holiday and days-since-holiday engineered features
* Warehouse-specific models or scaling
* Time-series cross-validation instead of single holdout
* Hyperparameter tuning (Optuna) or lightweight ARIMA ensembling

---

## Reproducibility Guidelines

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
cd forecast
```

### 3. Train Model

```bash
python train.py
```

This will save the trained model into `models/`.

### 4. Run inference

```bash
python inference.py --input data/forecast/test_forecast.csv
```

### 5. Evaluate results

```bash
python evaluation.py
```

### 6. Launch the Streamlit dashboard

```bash
streamlit run app.py
```
➡️ [Watch the demo](https://youtu.be/mU2ZH6Nc6Qk)
---

## Conclusion

This project delivers a **complete forecasting pipeline** for multi-warehouse demand prediction. It combines engineered features, gradient boosting, and a user-friendly dashboard to provide high-accuracy forecasts (sMAPE < 2%).

The pipeline is **fully reproducible** and structured for future improvements (warehouse-specific models, new feature engineering, advanced architectures).

