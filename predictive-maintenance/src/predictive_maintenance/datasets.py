# src/datasets.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class EngineSequenceDataset(Dataset):
    """Sliding-window sequences of sensor data with RUL labels."""
    def __init__(self, df, feature_cols, window_size=30):
        self.samples = []
        self.feature_cols = feature_cols
        self.window_size = window_size
        for engine_number in df["engine_number"].unique():
            engine_df = df[df["engine_number"] == engine_number]
            values = engine_df[feature_cols + ["RUL"]].values
            n_rows = values.shape[0]
            if n_rows > window_size:
                for start in range(n_rows - window_size):
                    end = start + window_size
                    X = values[start:end, :-1]
                    y = values[end-1, -1]
                    self.samples.append((X, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def make_dataloader(df, feature_cols, window_size=30, batch_size=64, shuffle=True):
    dataset = EngineSequenceDataset(df, feature_cols, window_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def make_test_windows(df, feature_cols, window_size=30):
    """Extract last window_size cycles per engine for test prediction."""
    X, engine_numbers = [], []
    for eng_id, group in df.groupby("engine_number"):
        group = group.sort_values("cycle")
        if len(group) >= window_size:
            X.append(group[feature_cols].iloc[-window_size:].values)
            engine_numbers.append(eng_id)
    X = np.array(X, dtype=np.float32)
    return torch.tensor(X), engine_numbers
