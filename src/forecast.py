# forecast_30_days.py
import pandas as pd
import numpy as np
import joblib
import os

def forecast_30_days():
    """Load ARIMA+GARCH models and forecast 30 days ahead"""
    
    # Paths to saved models
    arima_path = "models/arima_model.pkl"
    garch_path = "models/garch_model.pkl"
    forecast_path = "/opt/airflow/data/predictions.csv"

    # Load models
    arima_model = joblib.load(arima_path)
    garch_model = joblib.load(garch_path)

    # Forecast mean using ARIMA
    steps = 30
    arima_forecast = arima_model.forecast(steps=steps)

    # Forecast variance using GARCH
    # Using last residuals to initialize volatility forecast
    garch_forecast = garch_model.forecast(horizon=steps, method='simulation')
    garch_var_forecast = garch_forecast.variance.values[-1]  # last row of forecasted variances
    garch_vol_forecast = np.sqrt(garch_var_forecast)

    # Combine mean + volatility into a DataFrame
    forecast_dates = pd.date_range(start=pd.Timestamp.today(), periods=steps, freq='D')
    forecast_df = pd.DataFrame({
        "date": forecast_dates,
        "predicted_return": arima_forecast,
        "predicted_volatility": garch_vol_forecast
    })

    # Save for monitoring
    os.makedirs(os.path.dirname(forecast_path), exist_ok=True)
    forecast_df.to_csv(forecast_path, index=False)
    print(f"✅ Forecast saved to {forecast_path}")
