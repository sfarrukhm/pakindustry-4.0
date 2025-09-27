import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
import os
from utils import CombinedLoss,evaluate_metrics

class CNN_LSTM_RUL(nn.Module):
    def __init__(self, input_dim, hidden_units=[100, 50], dropout=0.2, conv_channels=[32, 64], kernel_size=5):
        super().__init__()

        # 1D CNN layers
        self.conv1 = nn.Conv1d(input_dim, conv_channels[0], kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(conv_channels[0])
        self.conv2 = nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(conv_channels[1])
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # LSTM (unidirectional only → practical deployment)
        self.lstm1 = nn.LSTM(conv_channels[1], hidden_units[0], batch_first=True, bidirectional=False)
        self.norm1 = nn.LayerNorm(hidden_units[0])
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(hidden_units[0], hidden_units[1], batch_first=True, bidirectional=False)
        self.norm2 = nn.LayerNorm(hidden_units[1])
        self.dropout2 = nn.Dropout(dropout)

        # Fully connected layer
        self.fc = nn.Linear(hidden_units[1], 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1)                # → (batch, input_dim, seq_len)
        x = self.relu(self.bn1(self.conv1(x)))  # → (batch, conv1, seq_len)
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))  # → (batch, conv2, seq_len/2)
        x = self.pool(x)
        x = x.permute(0, 2, 1)                # → (batch, seq_len_reduced, conv_channels[-1])

        # LSTM layers
        out, _ = self.lstm1(x)
        out = self.norm1(out)
        out = self.dropout1(out)

        out, _ = self.lstm2(out)
        out = self.norm2(out)
        out = self.dropout2(out)

        # Last timestep
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def train_lstm_model(train_loader, val_loader, input_dim, hidden_units=[100,50], dropout=0.2, alpha=0.5,
                     lr=0.001, epochs=60, patience=15, device="cuda", save_dir="./models"):
    
    os.makedirs(save_dir, exist_ok=True)

    model = CNN_LSTM_RUL(input_dim, hidden_units, dropout).to(device)
    criterion = CombinedLoss(alpha=0.8) 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

    def lr_lambda(step): return 0.9 ** (step / 1000)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_val_loss, patience_counter, best_model_state = np.inf, 0, None
    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": [], "val_r2": [], "val_nasa": []}

    pbar = tqdm(range(epochs), desc="Training", unit="epoch", ncols=100)

    for epoch in pbar:
        # --------------------
        # Training
        # --------------------
        model.train()
        train_losses = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        # --------------------
        # Validation + metrics
        # --------------------
        model.eval()
        val_losses, y_true_all, y_pred_all = [], [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device).unsqueeze(1)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_losses.append(loss.item())
                y_true_all.append(y)
                y_pred_all.append(outputs)

        train_loss, val_loss = np.mean(train_losses), np.mean(val_losses)
        y_true_all, y_pred_all = torch.cat(y_true_all), torch.cat(y_pred_all)
        mae, rmse, r2, nasa = evaluate_metrics(y_true_all, y_pred_all)

        # --------------------
        # History tracking
        # --------------------
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(mae)
        history["val_rmse"].append(rmse)
        history["val_r2"].append(r2)
        history["val_nasa"].append(nasa)

        # --------------------
        # Early stopping
        # --------------------
        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
            best_model_state = model.state_dict()
            torch.save(best_model_state, os.path.join(save_dir, "best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                tqdm.write("⏹️ Early stopping triggered.")
                break

        # --------------------
        # Logging
        # --------------------
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            tqdm.write(
                f"📊 Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.2f} | NASA: {nasa:.2f} | "
                f"Best Val: {best_val_loss:.4f}"
            )

    # --------------------
    # Finalize
    # --------------------
    torch.save(model.state_dict(), os.path.join(save_dir, "last.pt"))
    if best_model_state: model.load_state_dict(best_model_state)
    
    return model, history
