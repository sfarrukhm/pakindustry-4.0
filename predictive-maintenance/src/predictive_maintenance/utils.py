import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

import torch
import torch.nn as nn
import os

from src.predictive_maintenance.data import  make_dataloader, make_test_windows, load_data, add_engineered_features, scale
from src.predictive_maintenance.models import CNN_LSTM_RUL, train_lstm_model

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


def run_pipeline(train_path, test_path, rul_path, 
                 feature_cols,epochs, window_size=30, batch_size=64,lr=0.001,alpha=0.5,  
                 train_split=0.7, val_split=0.15, device="cuda",):

    # 1. Load data
    train_df, test_df, test_rul = load_data(train_path, test_path, rul_path)

    # 2. Feature engineering
    train_df = add_engineered_features(train_df)
    test_df = add_engineered_features(test_df)

    # 3. Scaling
    train_df, test_df = scale(train_df, test_df, feature_cols)

    # 4. Train/val/test split (based on engine_number)
    engine_numbers = train_df["engine_number"].unique()
    np.random.seed(42)
    np.random.shuffle(engine_numbers)

    n_train = int(train_split * len(engine_numbers))
    n_val = int(val_split * len(engine_numbers))

    train_ids = engine_numbers[:n_train]
    val_ids = engine_numbers[n_train:n_train+n_val]
    test_ids = engine_numbers[n_train+n_val:]

    train_df_split = train_df[train_df["engine_number"].isin(train_ids)]
    val_df_split   = train_df[train_df["engine_number"].isin(val_ids)]
    test_df_split  = train_df[train_df["engine_number"].isin(test_ids)]

    # 5. Build DataLoaders
    train_loader = make_dataloader(train_df_split, feature_cols, window_size, batch_size, shuffle=True)
    val_loader   = make_dataloader(val_df_split, feature_cols, window_size, batch_size, shuffle=False)
    test_loader  = make_dataloader(test_df_split, feature_cols, window_size, batch_size, shuffle=False)

    # 6. Train model
    input_dim = len(feature_cols)
    model, history = train_lstm_model(train_loader, val_loader, input_dim, lr=lr,alpha=alpha, epochs=epochs, device=device)

   

    ### ADDED: Load best model for evaluation (if exists)
    best_model_path = "./models/best.pth"
    if os.path.exists(best_model_path):
        best_model = LSTM_RUL(input_dim).to(device)
        best_model.load_state_dict(torch.load(best_model_path, map_location=device))
        best_model.eval()
        model = best_model  # overwrite model with best version
        print("✅ Loaded best model for evaluation")

    # 7. Evaluate on validation set
    print("\nEvaluate on validation set:")
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X).squeeze()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    print_score(y_true, y_pred, prefix="train")
    plot_scatter(y_true, y_pred, prefix="train")
    plot_histogram(y_true, y_pred, prefix="train")
    # 8. Evaluate on split test set
    print("\nEvaluate on split test set:")
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X).squeeze()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    print_score(y_true, y_pred)
    plot_scatter(y_true, y_pred)
    plot_histogram(y_true, y_pred)

    # 9. Evaluate on real test set (last windows only)
    print("\nEvaluate on real test set:")
    X_test, engine_numbers = make_test_windows(test_df, feature_cols, window_size)
    X_test = X_test.to(device)
    with torch.no_grad():
        y_pred = model(X_test).squeeze().cpu().numpy()
    y_true = np.array(test_rul[:len(engine_numbers)])  # ground truth

    print_score(y_true, y_pred)
    plot_scatter(y_true, y_pred)
    plot_histogram(y_true, y_pred)

    return model, history, y_pred, y_true

