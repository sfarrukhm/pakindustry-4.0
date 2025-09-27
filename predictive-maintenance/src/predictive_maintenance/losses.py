
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error

import torch
import torch.nn as nn
import os

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
    mape = mean_absolute_percentage_error(y_true, y_pred)

    # NASA score
    error = y_pred - y_true
    nasa = np.where(
        error < 0,
        np.exp(-error / 13) - 1,
        np.exp(error / 10) - 1
    ).sum()

    return mae, rmse, mape, nasa

