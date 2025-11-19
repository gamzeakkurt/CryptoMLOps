# generate_dashboard.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/opt/airflow/data/dashboard.png")
PRED = Path("/opt/airflow/data/predictions_with_uncertainty.csv")
ACTUAL = Path("/opt/airflow/data/actuals.csv")

def generate_dashboard():
    print("📊 Generating dashboard...")

    plt.figure(figsize=(12, 6))

    # --------------------
    # Load predictions
    # --------------------
    if not PRED.exists():
        print("⚠️ No prediction file found. Dashboard aborted.")
        return

    pred = pd.read_csv(PRED, parse_dates=["date"])
    pred = pred.drop_duplicates(subset=["date"]).sort_values("date")
    pred = pred.set_index("date")

    # Compute cumulative predictions
    pred["cum_pred"] = pred["predicted_return"].cumsum()

    # Plot predictions
    plt.plot(pred.index, pred["cum_pred"], label="Predicted cumulative return")

    # Confidence interval
    if "upper" in pred.columns and "lower" in pred.columns:
        plt.fill_between(
            pred.index,
            pred["lower"].cumsum(),
            pred["upper"].cumsum(),
            alpha=0.2,
            label="95% CI",
        )

    # --------------------
    # Load actuals (optional)
    # --------------------
    if ACTUAL.exists():
        act = pd.read_csv(ACTUAL, parse_dates=["date"])
        act = act.drop_duplicates(subset=["date"]).sort_values("date")
        act = act.set_index("date")

        # Align to prediction dates (safe)
        act = act.reindex(pred.index)

        # If column exists
        if "actual_return" in act.columns:
            act["cum_actual"] = act["actual_return"].fillna(0).cumsum()
        elif "close" in act.columns:
            act["cum_actual"] = act["close"].pct_change().fillna(0).cumsum()
        else:
            print("⚠️ Actuals exist but no usable column (actual_return or close).")
            act["cum_actual"] = None

        plt.plot(act.index, act["cum_actual"], label="Actual cumulative return")

    else:
        print("ℹ️ No actuals.csv found. Dashboard will show only predictions.")

    # --------------------
    # Finalize plot
    # --------------------
    plt.title("Crypto Forecast Dashboard")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT)
    plt.close()

    print(f"✅ Dashboard saved to {OUT}")
