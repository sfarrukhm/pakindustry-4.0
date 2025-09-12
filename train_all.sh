#!/bin/bash
# Train all models in sequence

echo "=== Training Defect Detection Models ==="
python src/cv/train.py

echo "=== Training Predictive Maintenance Models ==="
python src/pm/train.py

echo "=== Training Supply Chain Forecasting Models ==="
python src/forecast/train.py
