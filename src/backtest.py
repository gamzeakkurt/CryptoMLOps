# backtest.py
import pandas as pd
import numpy as np
from pathlib import Path
from data_pipeline import load_data, make_features
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

RESULT_PATH = Path("/opt/airflow/data/backtest_results.csv")

def backtest_model():
    print("🔎 Running ARIMA+GARCH walk-forward backtesting...")

    df = load_data()
    _, y = make_features(df)
    y = y.values  # raw returns as numpy array

    tscv = TimeSeriesSplit(n_splits=5)

    fold = 1
    mse_scores, mae_scores, vol_errors = [], [], []

    for train_idx, test_idx in tscv.split(y):
        y_train, y_test = y[train_idx], y[test_idx]

        # ----- Fit ARIMA on training data -----
        arima_model = ARIMA(y_train, order=(1, 0, 1))
        arima_res = arima_model.fit()

        # Mean forecast for test section
        y_pred = arima_res.forecast(steps=len(y_test))

        # ----- Fit GARCH on ARIMA residuals -----
        residuals = y_train - arima_res.fittedvalues
        garch = arch_model(residuals, p=1, q=1, vol='Garch')
        garch_res = garch.fit(disp="off")

        # Volatility forecast
        vol_pred = garch_res.forecast(horizon=len(y_test)).variance.values[-1]
        vol_pred = np.sqrt(vol_pred)  # convert variance → std dev

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        mse_scores.append(mse)
        mae_scores.append(mae)
        vol_errors.append(vol_pred.mean())

        print(f"Fold {fold}: MSE={mse:.6f}, MAE={mae:.6f}, Vol={vol_pred.mean():.6f}")
        fold += 1

    # Save backtesting results
    results = pd.DataFrame({
        "fold": list(range(1, len(mse_scores) + 1)),
        "mse": mse_scores,
        "mae": mae_scores,
        "vol_forecast_mean": vol_errors
    })

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULT_PATH, index=False)

    print(f"✅ Backtest saved to {RESULT_PATH}")
