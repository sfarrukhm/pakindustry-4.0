# Rohlik Orders Forecasting
Advanced supply chain demand forecasting using machine learning and time series analysis

## Executive Summary

This project tackles daily order forecasting across 7 European warehouses (Prague, Brno, Munich, Frankfurt, Budapest) using 3+ years of historical data from December 2020 to March 2024. We implemented a LightGBM-based solution with engineered lag features, rolling statistics, and calendar event integration to predict daily order volumes per warehouse. The model achieves exceptional performance with 1.97% sMAPE and 3.2% NRMSE, significantly exceeding the hackathon requirement of ≤10% forecast error. Final validation metrics: RMSE 208.40, MAE 126.87, sMAPE 1.97%, NRMSE 0.032.

## Repository Structure

```
forecast/
├── train.py              # Model training and feature engineering
├── inference.py          # Model inference on test data
├── evaluation.py         # Evaluation metrics from saved validation results
├── models/               # Saved trained models
├── results/              # Validation metrics and predictions
└── data/forecast/        # Raw dataset files
```

## Data Description

**Primary Files:**
- `train.csv` - Historical daily orders per warehouse (2020-12-05 to 2024-03-15)
- `train_calendar.csv` - Calendar event flags (holiday, shops_closed, school_holidays, shutdown, blackout, etc.)
- `test.csv` - Test set warehouse-date combinations for prediction

**Key Details:**
- **Target Variable:** Daily total orders per warehouse
- **Warehouses:** Prague_1, Brno_1, Prague_2, Prague_3, Munich_1, Frankfurt_1, Budapest_1
- **Calendar Integration:** Event flags merged on date to capture seasonal and operational patterns
- **Feature Availability:** Some features (user_activity_1/2, precipitation) exist only in training data and are excluded from modeling

## Model & Features

**Model Architecture:**
- Primary model: LightGBM gradient boosting regressor
- Validation approach: Last 14 days held out as validation set
- Reproducibility: Fixed seed (42) across all random operations

**Engineered Features:**
- **Temporal:** weekday, month, year, warehouse_weekday interactions
- **Lag Features:** 7, 14, 21, 28, 35-day historical order lags
- **Rolling Statistics:** 7-35 day rolling means for trend capture
- **Event Integration:** Holiday flags, shop closures, shutdowns, and operational events
- **Warehouse Encoding:** Categorical encoding for location-specific patterns

## Evaluation & Metrics

**Validation Performance:**
- **RMSE: 208.40** - Root mean square error measuring average prediction magnitude
- **MAE: 126.87** - Mean absolute error showing typical prediction deviation  
- **sMAPE: 1.97%** - Symmetric mean absolute percentage error for scale-independent accuracy
- **NRMSE: 0.032** - Normalized RMSE relative to mean order volume

**Performance Context:** While absolute errors (RMSE/MAE) may appear large, they represent excellent accuracy relative to daily order volumes of 5,000-8,500 across warehouses. An RMSE of 208 orders on volumes averaging 6,000-8,000 daily orders translates to ~2.5-3.5% typical error, which is exceptional for supply chain forecasting where 5-15% error rates are industry standard.

**Hackathon Criterion:** Supply chain forecast error ≤ 10% (percentage metric). **Current model significantly exceeds this requirement** with both sMAPE (1.97%) and NRMSE (3.2%) well below the threshold.

## How to Run
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

4. **Training:**
    ```bash
    python train.py
    ```

5. **Inference:**
    ```bash
    python inference.py
    ```

6. **Evaluation:**
    ```bash
    python evaluation.py
    ```

**Output Files:**
- `models/best_model.pkl` - Trained LightGBM model
- `results/val_metrics.json` - Validation metrics with schema: `{"rmse": float, "mae": float, "smape": float, "nrmse": float}`
- `results/val_predictions.csv` - Validation predictions with columns: `id,date,warehouse,orders_true,orders_pred`

## Reproducibility

**Required Packages:**
python3, pandas, numpy, scikit-learn, lightgbm, joblib

**Optional Packages:**
pmdarima, optuna (for advanced improvements)

**Seed Configuration (top of train.py):**
```python
import os, random, numpy as np
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
```

## Limitations & Cautions

- **Holiday Anomalies:** Model may underperform during unprecedented holiday patterns or new event types not seen in training data
- **Historical Dependence:** Performance relies on continuation of established demand patterns; major market shifts could impact accuracy
- **Feature Availability:** Several potentially valuable features (user activity, weather) are training-only and cannot be leveraged for predictions
- **Warehouse Scaling:** Different order volumes across warehouses (Prague_1 ~8,500 vs Budapest_1 ~5,500 daily orders) may benefit from location-specific modeling approaches

## Next Steps / Recommended Improvements

- **Enhanced Rolling Features:** Add rolling standard deviation, minimum, and maximum alongside existing means
- **Cyclical Encoding:** Implement sine/cosine transforms for weekday and month to better capture cyclical patterns
- **Holiday Engineering:** Create days-until-holiday and days-since-holiday features for improved event modeling
- **Warehouse-Specific Models:** Train separate models per warehouse or add warehouse-specific scaling
- **Time-Series Cross-Validation:** Replace single holdout with expanding window validation for more robust evaluation
- **Hyperparameter Optimization:** Quick Optuna tuning for LightGBM parameters and optional lightweight ARIMA ensemble

## Presentation Summary for Judges

We developed a machine learning solution to forecast daily orders across 7 European warehouses for Rohlik's supply chain optimization. Our approach combines LightGBM gradient boosting with carefully engineered features including historical lag patterns, rolling trends, and operational event calendars spanning over 3 years of data. Key features capture weekly seasonality, warehouse-specific patterns, and external factors like holidays and operational shutdowns. We validated using the most recent 14 days as a holdout set, achieving 1.97% symmetric mean absolute percentage error - dramatically exceeding the competition's 10% accuracy requirement. This level of precision enables reliable inventory planning, reduces stockouts and overstocking costs, and supports efficient warehouse operations across multiple countries. Immediate next steps include warehouse-specific modeling and integration of additional external data sources for even greater accuracy.

## Contact & Attribution

Team Industrial AI - Techathon 2025