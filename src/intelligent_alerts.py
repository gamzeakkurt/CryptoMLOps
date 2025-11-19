import pandas as pd
import numpy as np
from pathlib import Path

BACKTEST_PATH = Path("/opt/airflow/data/backtest_results.csv")
DRIFT_PATH = Path("/opt/airflow/data/drift_report.csv")
FORECAST_PATH = Path("/opt/airflow/data/prediction.csv")
ACTUAL_PATH = Path("/opt/airflow/data/actuals.csv")

def intelligent_alerts():
    print("\n🚨 Checking intelligent MLOps alerts...\n")

    alerts = []

    # -----------------------------
    # 1. Model degradation alerts
    # -----------------------------
    if BACKTEST_PATH.exists():
        bt = pd.read_csv(BACKTEST_PATH)
        last_mse = bt["mse"].iloc[-1]
        prev_mse = bt["mse"].iloc[-2] if len(bt) > 1 else last_mse

        if last_mse > prev_mse * 1.25:  # 25% degradation
            alerts.append(f"⚠️ Model MSE increased by > 25%! ({prev_mse:.6f} ➜ {last_mse:.6f})")

        if last_mse > 0.00003:
            alerts.append(f"⚠️ High MSE detected ({last_mse:.6f})")

    # -----------------------------
    # 2. Drift detection alerts
    # -----------------------------
    if DRIFT_PATH.exists():
        drift = pd.read_csv(DRIFT_PATH)
        if drift["psi"].max() > 0.1:
            alerts.append(f"⚠️ PSI drift detected (max PSI: {drift['psi'].max():.3f})")

        if (drift["ks_p_value"] < 0.05).any():
            alerts.append(f"⚠️ KS-test drift detected! Several features distribution changed.")

    # -----------------------------
    # 3. Volatility alerts (from GARCH)
    # -----------------------------
    high_vol = None
    if BACKTEST_PATH.exists() and "vol_forecast_mean" in bt.columns:
        high_vol = bt["vol_forecast_mean"].iloc[-1]
        if high_vol > 0.01:  # high volatility threshold
            alerts.append(f"⚠️ High market volatility detected (σ={high_vol:.4f})")

    # -----------------------------
    # 4. Forecast accuracy alerts
    # -----------------------------
    if FORECAST_PATH.exists() and ACTUAL_PATH.exists():
        pred = pd.read_csv(FORECAST_PATH)
        actual = pd.read_csv(ACTUAL_PATH)

        if len(pred) > 0 and len(actual) > 0:
            merged = pred.merge(actual, how="inner", on="date")
            if "forecast" in merged.columns and "close" in merged.columns:
                y_true = merged["close"]
                y_pred = merged["forecast"]
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

                if mape > 15:  # 15% MAPE threshold
                    alerts.append(f"⚠️ Forecast error spike detected (MAPE={mape:.2f}%)")

    # -----------------------------
    # 5. Missing or stale data alerts
    # -----------------------------
    for file, name in [(FORECAST_PATH, "Forecast"), (ACTUAL_PATH, "Actual data")]:
        if not file.exists():
            alerts.append(f"⚠️ Missing file: {name} not generated today.")

    # -----------------------------
    # Summary
    # -----------------------------
    if alerts:
        print("🚨 **ALERTS TRIGGERED:**")
        for a in alerts:
            print(" - " + a)
        print("\n❗ Action recommended.\n")
    else:
        print("✅ No alerts. System healthy.\n")
