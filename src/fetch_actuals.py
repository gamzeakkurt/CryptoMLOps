import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

ACTUAL_PATH = Path("/opt/airflow/data/actuals.csv")

def fetch_actual_btc_price():
    """Fetch latest BTC/USDT hourly data and compute last day's return"""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1d", "limit": 1}
    response = requests.get(url, params=params)
    data = response.json()[0]
    close_price = float(data[4])

    today = datetime.utcnow().strftime("%Y-%m-%d")
    new_row = pd.DataFrame([{"date": today, "close": close_price}])

    if ACTUAL_PATH.exists():
        df = pd.read_csv(ACTUAL_PATH)
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row

    df.to_csv(ACTUAL_PATH, index=False)
    print(f"✅ Saved actual BTC price for {today}")
