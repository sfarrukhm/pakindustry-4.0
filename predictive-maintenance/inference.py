# inference.py
import os
import torch
import pandas as pd
from src.predictive_maintenance.models import BiLSTM_GRU_RUL
from src.predictive_maintenance.data import (
    load_data,
    add_engineered_features,
    scale,
    create_feature_cols,
    make_test_windows
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/best.pt"
WINDOW_SIZE = 30
TRAIN_PATH="data/CMaps/train_FD001.txt"

def run_inference(input_file: str):
    """
    Run inference on input test data using trained CNN-LSTM model.
    Returns a DataFrame with predictions only (no true RUL).
    """

    # Infer directory structure
    data_dir = os.path.dirname(input_file)
    train_path = TRAIN_PATH # for scaling
    test_path = input_file

    # ===== Load data (ignore rul_path) =====
    train_df, test_df, _ = load_data(train_path, test_path, rul_path=None)

    # ===== Feature engineering + scaling =====
    test_df = add_engineered_features(test_df)
    train_df = add_engineered_features(train_df)

    feature_cols = create_feature_cols(train_path, test_path, rul_path=None, max_rul=125)
    train_df, test_df = scale(train_df, test_df, feature_cols)

    # ===== Load model =====
    input_dim = len(feature_cols)
    model = BiLSTM_GRU_RUL(input_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # ===== Prepare test windows =====
    X_test, engine_numbers = make_test_windows(test_df, feature_cols, window_size=WINDOW_SIZE)
    X_test = X_test.to(DEVICE)

    # ===== Inference =====
    with torch.no_grad():
        y_pred = model(X_test).squeeze().cpu().numpy()

    df_preds = pd.DataFrame({"engine_number": engine_numbers, "pred_RUL": y_pred})

    return df_preds
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference with trained BiLSTM_GRU_RUL model")
    parser.add_argument("--input", type=str, required=True, help="Path to input test file (CMAPSS test dataset)")
    args = parser.parse_args()

    predictions = run_inference(args.input)