import json

# Load validation results
with open("results/val_metrics.json", "r") as f:
    results = json.load(f)

print("===== Evaluation Results =====")
print(f"sMAPE: {results['smape']:.2f}%")
print(f"NRMSE: {results['nrmse']:.3f}")
print(f"MAE:   {results['mae']:.2f}")
print(f"RMSE:  {results['rmse']:.2f}")
