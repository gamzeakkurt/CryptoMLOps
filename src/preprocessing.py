import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller, acf, q_stat
from scipy.stats import shapiro


def preprocess_data(input_path="/opt/airflow/data/BTCUSDT_6months_hourly.csv",
                    output_path="/opt/airflow/data/preprocessed.csv"):
    
    print("🚀 Starting data preprocessing...")

    # --- Load data ---
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values("timestamp")

    # --- Basic log returns ---
    df['return'] = np.log(df['close']).diff()
    df = df.dropna()

    # --- Time-based features ---
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month

    # --- Rolling statistics ---
    for window in [3, 6, 12, 24]:
        df[f'rolling_mean_{window}'] = df['return'].rolling(window).mean()
        df[f'rolling_std_{window}'] = df['return'].rolling(window).std()
        df[f'rolling_volatility_{window}'] = df['return'].rolling(window).std() * np.sqrt(window)

    df = df.dropna()

    # --- Feature scaling ---
    feature_cols = ['return', 'hour', 'day_of_week', 'day_of_month', 'month'] + [
        col for col in df.columns if "rolling_" in col
    ]

    # --- Statistical tests ---
    stats_results = []

    # 1️⃣ ADF Test (stationarity)
    adf_result = adfuller(df['return'])
    stats_results.append({
        "test": "Augmented Dickey-Fuller",
        "statistic": adf_result[0],
        "p_value": adf_result[1],
        "stationary": adf_result[1] < 0.05
    })

    # 2️⃣ Ljung–Box Test (autocorrelation)
    acf_vals = acf(df['return'], fft=False, nlags=20)
    q_stat_vals, p_vals = q_stat(acf_vals[1:], len(df['return']))
    stats_results.append({
        "test": "Ljung–Box",
        "statistic": q_stat_vals[-1],
        "p_value": p_vals[-1],
        "autocorrelated": p_vals[-1] < 0.05
    })

    # 3️⃣ Shapiro–Wilk Test (normality)
    shapiro_stat, shapiro_p = shapiro(df['return'].sample(min(5000, len(df))))  # limit size
    stats_results.append({
        "test": "Shapiro–Wilk",
        "statistic": shapiro_stat,
        "p_value": shapiro_p,
        "normal_distribution": shapiro_p > 0.05
    })

    stats_df = pd.DataFrame(stats_results)
    stats_path = Path(output_path).parent / "statistical_tests.csv"
    stats_df.to_csv(stats_path, index=False)

    print("📊 Statistical tests:")
    print(stats_df)
    print(f"✅ Saved statistical test results to {stats_path}")

    # --- Save scaled dataset ---
    df.to_csv(output_path, index=False)
    print(f"✅ Preprocessed dataset saved to {output_path}")

    return df
