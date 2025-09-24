import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def print_score(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    errors = np.clip(y_pred - y_true, -50, 50)
    nasa_terms = [np.exp(-err/10) - 1 if err < 0 else np.exp(err/13) - 1 for err in errors]
    nasa_score = np.sum(nasa_terms)

    print(f"MAE       : {mae:.2f}")
    print(f"RMSE      : {rmse:.2f}")
    print(f"R²        : {r2:.2f}")
    print(f"NASA Score: {nasa_score:.2f}")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "NASA": nasa_score}

def plot_scatter(y_true, y_pred, save_path=None):
    plt.figure(figsize=(8,6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], 'r--')
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("True vs Predicted RUL")
    plt.grid()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_histogram(y_true, y_pred, save_path=None):
    errors = y_pred - y_true
    plt.figure(figsize=(10,6))
    sns.histplot(errors, bins=50, kde=True)
    plt.title("Distribution of Prediction Errors (Predicted - True RUL)")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
