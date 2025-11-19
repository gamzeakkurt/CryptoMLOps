# detect_regime.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

DATA_PATH = Path("/opt/airflow/data/BTCUSDT_6months_hourly.csv")
REGIME_PATH = Path("/opt/airflow/data/regimes.csv")

def detect_regime(window=24, n_clusters=2):
    """
    Detect market regimes using rolling volatility and clustering.
    """
    print("🔎 Detecting volatility regimes...")
    df = pd.read_csv(DATA_PATH)
    if "close" not in df.columns:
        print("⚠️ 'close' missing.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["return"] = np.log(df["close"]).diff()
    df = df.dropna()

    vol_series = df["return"].rolling(window).std().dropna()
    if len(vol_series) < n_clusters:
        print("⚠️ Not enough data for regime detection.")
        return

    X = vol_series.values.reshape(-1,1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    regimes = pd.DataFrame({
        "timestamp": df["timestamp"].iloc[window-1:].values,
        "volatility": vol_series.values,
        "regime": kmeans.labels_
    })
    regimes.to_csv(REGIME_PATH, index=False)
    print(f"✅ Saved regimes to {REGIME_PATH}")
