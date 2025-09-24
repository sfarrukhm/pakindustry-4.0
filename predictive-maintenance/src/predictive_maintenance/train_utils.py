import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import torch
import os
from src.predictive_maintenance.data import load_data, add_engineered_features, scale
from src.predictive_maintenance.datasets import make_dataloader, make_test_windows
from src.predictive_maintenance.models import LSTM_RUL, train_lstm_model
from src.predictive_maintenance.metrics import plot_histogram, plot_scatter, print_score

def run_pipeline(train_path, test_path, rul_path, 
                 feature_cols,epochs, window_size=30, batch_size=64, 
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
    model, history = train_lstm_model(train_loader, val_loader, input_dim, epochs=epochs, device=device)

   

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
    print_score(y_true, y_pred)
    plot_scatter(y_true, y_pred)
    plot_histogram(y_true, y_pred)

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

