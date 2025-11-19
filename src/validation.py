# validation.py
import pandas as pd
from pathlib import Path

DATA_PATH = Path("/opt/airflow/data/BTCUSDT_6months_hourly.csv")
REPORT_PATH = Path("/opt/airflow/data/data_validation_report.csv")

def validate_data():
    """
    Simple data validation checks:
    - existence, shape, timestamp gaps, nulls, duplicates
    """
    print("🔎 Running data validation...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file missing: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    report = {}

    report["n_rows"] = len(df)
    report["n_columns"] = len(df.columns)
    report["n_nulls"] = int(df.isnull().sum().sum())
    report["n_duplicates"] = int(df.duplicated().sum())

    # timestamp gap check (if timestamp column exists)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        diffs = df["timestamp"].sort_values().diff().dropna()
        report["median_gap_seconds"] = diffs.dt.total_seconds().median()
        report["max_gap_seconds"] = diffs.dt.total_seconds().max()

    pd.DataFrame([report]).to_csv(REPORT_PATH, index=False)
    print(f"✅ Data validation written to {REPORT_PATH}")
