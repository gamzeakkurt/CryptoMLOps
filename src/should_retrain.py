# should_retrain.py
import pandas as pd
from pathlib import Path

DRIFT = Path("/opt/airflow/data/drift_report.csv")
BACKTEST = Path("/opt/airflow/data/backtest_results.csv")
DECISION = Path("/opt/airflow/data/should_retrain_flag.txt")

def should_retrain(mse_threshold=1e-4, psi_threshold=0.2):
    """
    Decide whether to retrain based on drift and backtest.
    Writes a flag file with YES/NO.
    """
    print("🧠 Deciding whether to retrain...")
    should = False
    reasons = []

    if DRIFT.exists():
        df = pd.read_csv(DRIFT)
        if (df["psi"] > psi_threshold).any() or (df["ks_p_value"] < 0.05).any():
            should = True
            reasons.append("drift")

    if BACKTEST.exists():
        b = pd.read_csv(BACKTEST)
        if b["mse"].mean() > mse_threshold:
            should = True
            reasons.append("backtest_mse_high")

    DECISION.write_text("YES" if should else "NO")
    print(f"✅ Should retrain: {should} (reasons: {reasons})")
