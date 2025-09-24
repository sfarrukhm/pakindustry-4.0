# src/data.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(train_path, test_path, rul_path, max_rul=125):
    """Load and preprocess NASA C-MAPSS FD001 dataset with capped RUL."""
    col_names = (
        ["engine_number", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] +
        [f"sensor_{i}" for i in range(1, 22)]
    )

    train_df = pd.read_csv(train_path, sep="\s+", header=None)
    test_df = pd.read_csv(test_path, sep="\s+", header=None)
    train_df = train_df.iloc[:, :len(col_names)]
    test_df  = test_df.iloc[:, :len(col_names)]
    train_df.columns = col_names
    test_df.columns  = col_names

    test_rul = pd.read_csv(rul_path, sep="\s+", header=None).iloc[:, 0].clip(upper=max_rul)

    # Compute RUL for training
    rul = train_df.groupby("engine_number")["cycle"].max().reset_index()
    rul.columns = ["engine_number", "max_cycle"]
    train_df = train_df.merge(rul, on="engine_number")
    train_df["RUL"] = train_df["max_cycle"] - train_df["cycle"]
    train_df["RUL"] = train_df["RUL"].clip(upper=max_rul)
    train_df.drop("max_cycle", axis=1, inplace=True)

    return train_df, test_df, test_rul


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features from selected sensors."""
    data = df.copy()
    # Thermal effects
    data["avg_temp"] = (data["sensor_2"] + data["sensor_3"] + data["sensor_4"]) / 3
    data["heat_to_fuel"] = (data["sensor_3"] + data["sensor_4"]) / data["sensor_12"].replace(0, np.finfo(float).eps)
    data["log_fuel_flow"] = np.log1p(data["sensor_12"])
    # Pressure
    data["pressure_range"] = data["sensor_11"] - data["sensor_6"]
    data["hpc_pressure_norm"] = data["sensor_7"] / data["sensor_11"].replace(0, np.finfo(float).eps)
    # Mechanical/rotational
    data["mech_energy"] = data["sensor_8"]**2 + data["sensor_9"]**2
    data["speed_err_fan"] = data["sensor_8"] - data["sensor_13"]
    data["speed_err_core"] = data["sensor_9"] - data["sensor_14"]
    # Cooling
    data["cooling_total"] = data["sensor_20"] + data["sensor_21"]
    data["cooling_per_bypass"] = data["cooling_total"] / data["sensor_15"].replace(0, np.finfo(float).eps)
    return data


def scale(df_train, df_test, feature_cols):
    """Scale features using StandardScaler for each operating condition."""
    df_train = df_train.copy()
    df_test = df_test.copy()
    op_cols = [c for c in df_train.columns if c.startswith("op_setting")]
    df_train["op_hash"] = df_train[op_cols].round(3).astype(str).agg("_".join, axis=1)
    df_test["op_hash"] = df_test[op_cols].round(3).astype(str).agg("_".join, axis=1)
    
    scalers = {}
    for op_id, group in df_train.groupby("op_hash"):
        scaler = StandardScaler()
        scaler.fit(group[feature_cols])
        scalers[op_id] = scaler
        df_train.loc[group.index, feature_cols] = scaler.transform(group[feature_cols])
    
    global_scaler = StandardScaler()
    global_scaler.fit(df_train[feature_cols])
    
    for op_id, group in df_test.groupby("op_hash"):
        scaler = scalers.get(op_id, global_scaler)
        df_test.loc[group.index, feature_cols] = scaler.transform(group[feature_cols])
    
    return df_train, df_test


def create_feature_cols(train_path: str, test_path: str, rul_path: str, max_rul: int = 125):
    """Determine final feature columns after loading, engineering, and dropping constants."""
    from .data import load_data, add_engineered_features

    train_df, test_df, _ = load_data(train_path, test_path, rul_path, max_rul)
    train_df = add_engineered_features(train_df)
    test_df = add_engineered_features(test_df)

    constant_features = train_df.columns[train_df.nunique() <= 2].tolist()
    exclude_cols = ["engine_number", "cycle", "RUL"]
    drop_cols = set(constant_features + exclude_cols)
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    print(f"Final feature columns ({len(feature_cols)}): {feature_cols}")
    return feature_cols
