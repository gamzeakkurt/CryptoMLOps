import pandas as pd
import numpy as np
import mlflow
from datetime import datetime

PRED_PATH = "/opt/airflow/data/predictions.csv"
ACTUAL_PATH = "/opt/airflow/data/actuals.csv"

def monitor_model_performance():
    """Compare actual vs predicted daily returns and log metrics."""
    preds = pd.read_csv(PRED_PATH)
    actuals = pd.read_csv(ACTUAL_PATH)

    merged = pd.merge(preds, actuals, on="date", how="inner")

    if len(merged) == 0:
        print("⚠️ No matching dates between predictions and actuals yet.")
        return

    merged["error"] = merged["predicted_return"] - merged["actual_return"]
    mse = np.mean(merged["error"] ** 2)
    mae = np.mean(np.abs(merged["error"]))

    mlflow.set_experiment("crypto_forecasting_monitoring")
    with mlflow.start_run(run_name=f"monitor_{datetime.utcnow().isoformat()}"):
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_param("n_samples", len(merged))

    print(f"📊 Monitoring logged to MLflow: MSE={mse:.6f}, MAE={mae:.6f}")
