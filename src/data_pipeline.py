import pandas as pd
import numpy as np
from pathlib import Path

def load_data(path='data/BTCUSDT_6months_hourly.csv'):

    print("Fetching and preparing data...")
    # Handle both absolute and relative paths
    if not Path(path).is_absolute():
        # If relative path, try relative to current working directory first
        # If that fails, try relative to this file's directory
        if not Path(path).exists():
            script_dir = Path(__file__).parent
            path = script_dir / path
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['return'] = np.log(df['close']).diff()
    df = df.dropna()
    return df

def make_features(df, window=24):
    for i in range(1, window + 1):
        df[f'return_lag_{i}'] = df['return'].shift(i)
    df = df.dropna()
    X = df[[f'return_lag_{i}' for i in range(1, window + 1)]]
    y = df['return']
    return X, y
