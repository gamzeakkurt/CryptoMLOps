# blend_models.py
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

PRED_OUT = Path("/opt/airflow/data/blended_predictions.csv")
RF_MODEL = Path("/opt/airflow/models/random_forest_model.pkl")
ARIMA_MODEL = Path("/opt/airflow/models/best_arima_model.pkl")
LSTM_MODEL = Path("/opt/airflow/models/lstm_model.h5")  # optional

def blend_models():
    """
    Load available model predictions (RF uses X, ARIMA uses series).
    Create an ensemble by averaging available model forecasts.
    """
    print("🔀 Running model blending...")
    # Load X and data
    X_path = Path("/opt/airflow/data/X.csv")
    df = None
    if X_path.exists():
        df = pd.read_csv(X_path)
    else:
        print("⚠️ X not available for RF blending.")

    preds = {}
    # RandomForest one-step predictions (if available)
    if RF_MODEL.exists() and df is not None:
        rf = joblib.load(RF_MODEL)
        preds["rf"] = rf.predict(df)

    # ARIMA: use saved forecast file if exists
    arima_forecast_file = Path("/opt/airflow/data/forecast_arima_garch.csv")
    if arima_forecast_file.exists():
        af = pd.read_csv(arima_forecast_file)
        preds["arima"] = af["arima_mean_forecast"].values

    # Build blended frame (align lengths)
    if preds:
        # pick the minimum length
        min_len = min(len(v) for v in preds.values())
        blended = np.mean([v[:min_len] for v in preds.values()], axis=0)
        out_df = pd.DataFrame({"prediction": blended})
        out_df.to_csv(PRED_OUT, index=False)
        print(f"✅ Blended predictions saved to {PRED_OUT}")
    else:
        print("⚠️ No model outputs found to blend.")
