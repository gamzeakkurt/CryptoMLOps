# train.py
import mlflow
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.metrics import mean_squared_error
import os
from data_pipeline import load_data, make_features
import joblib

def train_model():

    mlflow.set_experiment("crypto_forecasting_arima_garch_tuned")
    print("🔍 Starting ARIMA + GARCH Hyperparameter Tuning...")

    df = load_data()
    _, y = make_features(df)
    y = y.values


    #search space
    arima_params = [
        (1, 0, 1), (2, 0, 2), (3, 0, 1), (1, 1, 1)
    ]
    garch_params = [
        (1, 1), (2, 1), (1, 2)
    ]

    best_mse = float("inf")
    best_combo = None
    best_arima_model = None
    best_garch_model = None

    # --------------------------
    # GRID SEARCH
    # --------------------------
    for ar_order in arima_params:
        for (p, q) in garch_params:
            
            with mlflow.start_run(nested=True):
                mlflow.log_param("arima_order", ar_order)
                mlflow.log_param("garch_p", p)
                mlflow.log_param("garch_q", q)

                try:
                    # Fit ARIMA
                    arima_model = ARIMA(y, order=ar_order)
                    arima_res = arima_model.fit()

                    y_arima_pred = arima_res.predict(start=0, end=len(y)-1)
                    residuals = y - y_arima_pred

                    # Fit GARCH
                    garch_model = arch_model(residuals, vol="Garch", p=p, q=q, dist="normal")
                    garch_res = garch_model.fit(disp="off")

                    # Evaluate
                    mse = mean_squared_error(y, y_arima_pred)
                    mlflow.log_metric("mse", mse)

                    print(f"ARIMA{ar_order} + GARCH({p},{q}) → MSE: {mse:.8f}")

                    # Track best model
                    if mse < best_mse:
                        best_mse = mse
                        best_combo = (ar_order, p, q)
                        best_arima_model = arima_res
                        best_garch_model = garch_res

                except Exception as e:
                    print(f"❌ Failed for ARIMA{ar_order}+GARCH({p},{q}): {str(e)}")

    print("\n🎯 BEST MODEL FOUND")
    print(f"ARIMA{best_combo[0]} + GARCH({best_combo[1]}, {best_combo[2]})")
    print(f"Best MSE: {best_mse:.8f}")

    # --------------------------
    # SAVE BEST MODELS
    # --------------------------
    MODEL_DIR = "/opt/airflow/models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    arima_path = f"{MODEL_DIR}/best_arima_model.pkl"
    garch_path = f"{MODEL_DIR}/best_garch_model.pkl"

    joblib.dump(best_arima_model, arima_path)
    joblib.dump(best_garch_model, garch_path)

    print(f"✅ Best models saved: {arima_path}, {garch_path}")

    # --------------------------
    # LOG WINNING MODEL TO MLFLOW
    # --------------------------
    with mlflow.start_run():
        mlflow.log_param("best_arima_order", best_combo[0])
        mlflow.log_param("best_garch_p", best_combo[1])
        mlflow.log_param("best_garch_q", best_combo[2])
        mlflow.log_metric("best_mse", best_mse)
        mlflow.log_artifact(arima_path)
        mlflow.log_artifact(garch_path)


if __name__ == "__main__":
    train_model()
