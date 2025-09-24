# src/train_utils.py
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def print_score(y_true, y_pred):
    y_true = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
    y_pred = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    errors = np.clip(y_pred - y_true, -50, 50)
    nasa_terms = [np.exp(-err/10) - 1 if err < 0 else np.exp(err/13) - 1 for err in errors]
    nasa_score = np.sum(nasa_terms)
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}, NASA: {nasa_score:.2f}")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "NASA": nasa_score}


def plot_scatter(y_true, y_pred, save_path=None):
    y_true = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
    y_pred = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred
    plt.figure(figsize=(8,6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], 'r--')
    plt.xlabel("True RUL"); plt.ylabel("Predicted RUL"); plt.title("True vs Predicted RUL"); plt.grid()
    if save_path: plt.savefig(save_path, dpi=100, bbox_inches="tight"); plt.close()
    else: plt.show()


def plot_histogram(y_true, y_pred, save_path=None):
    y_true = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
    y_pred = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred
    errors = y_pred - y_true
    plt.figure(figsize=(10,6))
    sns.histplot(errors, bins=50, kde=True)
    plt.xlabel("Prediction Error"); plt.ylabel("Frequency"); plt.title("Prediction Error Distribution")
    if save_path: plt.savefig(save_path, dpi=100, bbox_inches="tight"); plt.close()
    else: plt.show()


def train_lstm_model(train_loader, val_loader, input_dim, epochs=50, device="cuda", lr=0.001, patience=10):
    from .models import LSTMModel, save_model

    model = LSTMModel(input_dim).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss, best_model = float('inf'), None
    history = {"train_loss": [], "val_loss": []}
    wait = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_loss = criterion(model(X), y)
                val_losses.append(val_loss.item())
        val_loss = np.mean(val_losses)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            save_model(model, "./models/pm/best.pth")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered")
                break

    return model, history
