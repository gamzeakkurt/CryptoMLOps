import json
from pathlib import Path
import pandas as pd
import numpy as np

BACKTEST_PATH = Path("/opt/airflow/data/backtest_results.csv")
DRIFT_PATH = Path("/opt/airflow/data/drift_report.csv")
FORECAST_PATH = Path("/opt/airflow/data/prediction.csv")
ACTUAL_PATH = Path("/opt/airflow/data/actuals.csv")

ALERT_PATH = Path("/opt/airflow/data/alerts.json")


def local_alerts():
    print("\n🛑 Running local alert checks...\n")
    alerts = []

    # -------------------------
    # 1. Model performance alerts
    # -------------------------
    if BACKTEST_PATH.exists():
        bt = pd.read_csv(BACKTEST_PATH)
        if len(bt) >= 2:
            last_mse = bt["mse"].iloc[-1]
            prev_mse = bt["mse"].iloc[-2]

            if last_mse > prev_mse * 1.25:
                alerts.append({
                    "type": "model_degradation",
                    "message": f"MSE increased 25%: {prev_mse:.6f} → {last_mse:.6f}"
                })

            if last_mse > 0.00003:
                alerts.append({
                    "type": "high_mse",
                    "message": f"High model MSE detected: {last_mse:.6f}"
                })

    # -------------------------
    # 2. Drift alerts
    # -------------------------
    if DRIFT_PATH.exists():
        drift = pd.read_csv(DRIFT_PATH)

        if drift["psi"].max() > 0.1:
            alerts.append({
                "type": "data_drift",
                "message": f"PSI drift detected (max={drift['psi'].max():.3f})"
            })

        if (drift["ks_p_value"] < 0.05).any():
            alerts.append({
                "type": "data_drift_ks",
                "message": "KS-test drift detected."
            })

    # -------------------------
    # 3. Forecast accuracy alerts
    # -------------------------
    if FORECAST_PATH.exists() and ACTUAL_PATH.exists():
        pred = pd.read_csv(FORECAST_PATH)
        actual = pd.read_csv(ACTUAL_PATH)

        merged = pred.merge(actual, how="inner", on="date")

        if "forecast" in merged and "close" in merged:
            y_true = merged["close"]
            y_pred = merged["forecast"]
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

            if mape > 15:
                alerts.append({
                    "type": "forecast_accuracy",
                    "message": f"High forecast error: MAPE={mape:.2f}%"
                })

    # -------------------------
    # 4. Missing files
    # -------------------------
    for f in [FORECAST_PATH, ACTUAL_PATH]:
        if not f.exists():
            alerts.append({
                "type": "missing_file",
                "message": f"Missing data file: {f.name}"
            })

    # -------------------------
    # Save alerts locally
    # -------------------------
    ALERT_PATH.write_text(json.dumps(alerts, indent=4))
    print(f"📁 Alerts saved to: {ALERT_PATH}")

    # -------------------------
    # Print into Airflow logs
    # -------------------------
    if alerts:
        print("\n🚨 ALERTS TRIGGERED:")
        for alert in alerts:
            print(f" - [{alert['type']}] {alert['message']}")
    else:
        print("✅ No alerts. System healthy.")

    return alerts
