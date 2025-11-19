# explainability.py
import pandas as pd
from pathlib import Path
import joblib
import shap
import matplotlib.pyplot as plt

X_PATH = Path("/opt/airflow/data/X.csv")
RF_MODEL = Path("/opt/airflow/models/random_forest_model.pkl")
OUT = Path("/opt/airflow/data/shap_summary.png")

def explain_model(n_samples=200):
    """Compute SHAP for tree models (RandomForest)."""
    print("🧭 Running explainability (SHAP)...")
    if not X_PATH.exists() or not RF_MODEL.exists():
        print("⚠️ Missing X or RF model for SHAP")
        return

    X = pd.read_csv(X_PATH)
    model = joblib.load(RF_MODEL)
    # sample to reduce compute
    Xs = X.sample(min(len(X), n_samples), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xs)
    shap.summary_plot(shap_values, Xs, show=False)
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT)
    plt.close()
    print(f"✅ SHAP summary saved to {OUT}")
