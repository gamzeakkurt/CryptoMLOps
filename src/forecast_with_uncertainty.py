# forecast_with_uncertainty.py
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

ARIMA_MODEL = Path("/opt/airflow/models/best_arima_model.pkl")
GARCH_MODEL = Path("/opt/airflow/models/best_garch_model.pkl")
OUT = Path("/opt/airflow/data/predictions_with_uncertainty.csv")

def forecast_with_uncertainty(steps=30):
    print("🔮 Generating forecast with uncertainty (ARIMA+GARCH)...")
    if not ARIMA_MODEL.exists() or not GARCH_MODEL.exists():
        print("⚠️ Models missing.")
        return

    arima = joblib.load(ARIMA_MODEL)
    garch = joblib.load(GARCH_MODEL)

    mean_forecast = arima.forecast(steps=steps)
    garch_fore = garch.forecast(horizon=steps)
    var_fore = garch_fore.variance.values[-1]
    vol_fore = np.sqrt(var_fore)

    dates = pd.date_range(start=pd.Timestamp.today(), periods=steps, freq='D')
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "predicted_return": np.array(mean_forecast),
        "predicted_volatility": vol_fore
    })
    df["upper"] = df["predicted_return"] + 1.96 * df["predicted_volatility"]
    df["lower"] = df["predicted_return"] - 1.96 * df["predicted_volatility"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"✅ Predictions with uncertainty saved to {OUT}")
