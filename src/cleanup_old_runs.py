# cleanup_old_runs.py
import os
from pathlib import Path
import time

MLRUNS = Path("/mlruns")  # if mlflow mounted here; adapt to your mlruns mount
ARCHIVE = Path("/opt/airflow/data/old_runs")

def cleanup_old_runs(keep_days=30):
    print("🧹 Cleaning old artifacts...")
    now = time.time()
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    # basic: move old model files from /opt/airflow/models older than keep_days
    models_dir = Path("/opt/airflow/models")
    for f in models_dir.glob("*"):
        if now - f.stat().st_mtime > keep_days * 86400:
            dest = ARCHIVE / f.name
            f.rename(dest)
            print(f"Moved {f} -> {dest}")
    print("✅ Cleanup complete.")
