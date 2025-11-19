# detect_anomalies.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest

DATA_PATH = Path("/opt/airflow/data/BTCUSDT_6months_hourly.csv")
ANOMALY_PATH = Path("/opt/airflow/data/anomalies.csv")

def detect_anomalies():
    """
    Detect anomalies using simple z-score on returns; fallback to IsolationForest.
    """
    print("🔍 Running anomaly detection...")
    df = pd.read_csv(DATA_PATH)
    if "close" not in df.columns:
        print("⚠️ 'close' column missing; skipping anomaly detection.")
        return

    df["return"] = np.log(df["close"]).diff()
    df = df.dropna().reset_index(drop=True)
    if df.empty:
        print("⚠️ No data after differencing.")
        return

    # z-score
    df["z"] = (df["return"] - df["return"].mean()) / (df["return"].std() + 1e-12)
    df["anomaly_z"] = df["z"].abs() > 3.5
    anomalies = df[df["anomaly_z"]].copy()

    # if none by zscore, try isolation forest
    if anomalies.empty:
        try:
            iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
            df["iso"] = iso.fit_predict(df[["return"]].fillna(0))
            anomalies = df[df["iso"] == -1].copy()
        except Exception as e:
            print("⚠️ IsolationForest failed:", e)

    if not anomalies.empty:
        anomalies.to_csv(ANOMALY_PATH, index=False)
        print(f"✅ Found {len(anomalies)} anomalies, saved to {ANOMALY_PATH}")
    else:
        print("✅ No anomalies detected.")
