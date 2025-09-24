import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import torch
import torch.nn as nn
import os

import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def print_score(y_true, y_pred, prefix=""):
    """
    Compute regression metrics and NASA scoring function.
    Works with numpy arrays or torch tensors.
    Saves metrics to results/metrics_{prefix}.txt if prefix is given.
    """
    if not isinstance(y_true, np.ndarray):
        y_true = y_true.detach().cpu().numpy()
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.detach().cpu().numpy()
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # NASA scoring function (punishes late predictions more)
    errors = np.clip(y_pred - y_true, -50, 50)
    nasa_terms = [np.exp(-err/10) - 1 if err < 0 else np.exp(err/13) - 1 for err in errors]
    nasa_score = np.sum(nasa_terms)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "NASA": nasa_score
    }

    # Print neatly
    print(f"MAE       : {mae:.2f}")
    print(f"RMSE      : {rmse:.2f}")
    print(f"R²        : {r2:.2f}")
    print(f"NASA Score: {nasa_score:.2f}")

    # Save metrics
    if prefix:
        path = os.path.join(RESULTS_DIR, f"metrics_{prefix}.txt")
        with open(path, "w") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")
        print(f"✅ Saved metrics to {path}")

    return metrics



def plot_scatter(y_true, y_pred, prefix=""):
    """
    Scatter plot of true vs predicted RUL.
    Saves to results/scatter_{prefix}.png if prefix is given.
    """
    if not isinstance(y_true, np.ndarray):
        y_true = y_true.detach().cpu().numpy()
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.detach().cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], 'r--')
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("True vs Predicted RUL")
    plt.grid()

    if prefix:
        path = os.path.join(RESULTS_DIR, f"scatter_{prefix}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved scatter plot to {path}")
    plt.close()


def plot_histogram(y_true, y_pred, prefix=""):
    """
    Histogram of prediction errors (Predicted - True).
    Saves to results/hist_{prefix}.png if prefix is given.
    """
    if not isinstance(y_true, np.ndarray):
        y_true = y_true.detach().cpu().numpy()
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.detach().cpu().numpy()

    errors = y_pred - y_true
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=50, kde=True)
    plt.title("Distribution of Prediction Errors (Predicted - True RUL)")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")

    if prefix:
        path = os.path.join(RESULTS_DIR, f"hist_{prefix}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved histogram to {path}")
    plt.close()

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta_under=13, beta_over=10):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.beta_under = beta_under
        self.beta_over = beta_over

    def nasa_loss(self, y_pred, y_true):
        error = y_pred - y_true
        loss = torch.where(
            error < 0,
            torch.exp(-error / self.beta_under) - 1,
            torch.exp(error / self.beta_over) - 1
        )
        return torch.mean(loss)

    def forward(self, y_pred, y_true):
        mse_loss = self.mse(y_pred, y_true)
        nasa_loss = self.nasa_loss(y_pred, y_true)
        return self.alpha * mse_loss + (1 - self.alpha) * nasa_loss


def evaluate_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(root_mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # NASA score
    error = y_pred - y_true
    nasa = np.where(
        error < 0,
        np.exp(-error / 13) - 1,
        np.exp(error / 10) - 1
    ).sum()

    return mae, rmse, r2, nasa