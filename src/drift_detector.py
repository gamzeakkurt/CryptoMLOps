# drift_detector.py
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ks_2samp

CURRENT = Path("/opt/airflow/data/X.csv")
REFERENCE = Path("/opt/airflow/data/reference_X.csv")
REPORT = Path("/opt/airflow/data/drift_report.csv")

def _psi(expected, actual, buckets=10):
    """Population Stability Index (simple implementation)."""
    exp_perc, act_perc = np.histogram(expected, bins=buckets)[0], np.histogram(actual, bins=buckets)[0]
    exp_perc = exp_perc / (exp_perc.sum() + 1e-12)
    act_perc = act_perc / (act_perc.sum() + 1e-12)
    # avoid zeros
    exp_perc = np.where(exp_perc == 0, 1e-6, exp_perc)
    act_perc = np.where(act_perc == 0, 1e-6, act_perc)
    return np.sum((exp_perc - act_perc) * np.log(exp_perc / act_perc))

def drift_detector():
    print("🔍 Running drift detection (PSI + KS)...")
    if not CURRENT.exists():
        raise FileNotFoundError("Current X missing")
    cur = pd.read_csv(CURRENT)
    if not REFERENCE.exists():
        cur.to_csv(REFERENCE, index=False)
        print("ℹ️ Reference created; no drift this run.")
        return

    ref = pd.read_csv(REFERENCE)
    rows = []
    for col in cur.columns:
        try:
            psi_val = _psi(ref[col].values, cur[col].values)
            ks_stat, ks_p = ks_2samp(ref[col].dropna(), cur[col].dropna())
            rows.append({
                "feature": col,
                "psi": float(psi_val),
                "ks_statistic": float(ks_stat),
                "ks_p_value": float(ks_p),
                "psi_drift": psi_val > 0.2,
                "ks_drift": ks_p < 0.05
            })
        except Exception as e:
            print(f"⚠️ drift check failed for {col}: {e}")

    pd.DataFrame(rows).to_csv(REPORT, index=False)
    print(f"✅ Drift report saved to {REPORT}")
