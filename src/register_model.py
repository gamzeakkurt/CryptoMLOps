# register_model.py
import mlflow
from pathlib import Path

MODEL_DIR = Path("/opt/airflow/models")
ARIMA = MODEL_DIR / "best_arima_model.pkl"
GARCH = MODEL_DIR / "best_garch_model.pkl"

def register_model():
    """
    Simple MLflow registration stub: logs artifacts and prints model info.
    For a real registry, call mlflow.register_model(...)
    """
    print("📦 Registering model to MLflow (artifact only)...")
    if not ARIMA.exists() or not GARCH.exists():
        print("⚠️ No models to register.")
        return

    with mlflow.start_run():
        mlflow.log_artifact(str(ARIMA))
        mlflow.log_artifact(str(GARCH))
    print("✅ Models logged to MLflow. For full registry use mlflow.register_model.")
