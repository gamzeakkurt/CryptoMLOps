from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
sys.path.append("/opt/airflow/src")
from data_pipeline import load_data, make_features
from validation import validate_data
from preprocessing import preprocess_data
from detect_anomalies import detect_anomalies
from detect_regime import detect_regime
from train import train_model
from backtest import backtest_model
from drift_detector import drift_detector
from should_retrain import should_retrain
from forecast_with_uncertainty import forecast_with_uncertainty
from generate_dashboard import generate_dashboard
from notify_team import notify_team
from detect_anomalies import detect_anomalies
from cleanup_old_runs import cleanup_old_runs
from explainability import explain_model
from blend_models import blend_models
from register_model import register_model
from intelligent_alerts import intelligent_alerts


default_args = {
    "owner": "crypto_team",
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

def fetch_and_process_data():
    df = load_data()
    X, y = make_features(df)
    X.to_csv("/opt/airflow/data/X.csv", index=False)
    y.to_csv("/opt/airflow/data/y.csv", index=False)

with DAG(
    dag_id="crypto_forecast_monitoring_pipeline",
    default_args=default_args,
    description="Train, forecast, fetch actuals, and monitor model performance",
    schedule_interval="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:
    t_valid = PythonOperator(task_id="validate_data", python_callable=validate_data)
   
    t_preproc = PythonOperator(task_id="preprocess_data", python_callable=preprocess_data)
    t_anom = PythonOperator(task_id="detect_anomalies", python_callable=detect_anomalies)
    t_regime = PythonOperator(task_id="detect_regime", python_callable=lambda: detect_regime(window=24, n_clusters=2))
    t_train = PythonOperator(task_id="train_model", python_callable=train_model)
    t_backtest = PythonOperator(task_id="backtest_model", python_callable=backtest_model)
    t_drift = PythonOperator(task_id="drift_detection", python_callable=drift_detector)
    t_should = PythonOperator(task_id="should_retrain", python_callable=should_retrain)

    t_explain = PythonOperator(task_id="explain_model", python_callable=explain_model)
    t_blend = PythonOperator(task_id="blend_models", python_callable=blend_models)
    t_forecast = PythonOperator(task_id="forecast_uncertainty", python_callable=lambda: forecast_with_uncertainty(steps=30))
    t_dashboard = PythonOperator(task_id="generate_dashboard", python_callable=generate_dashboard)
    t_register = PythonOperator(task_id="register_model", python_callable=register_model)
    t_alert = PythonOperator(task_id="intelligent_alerts", python_callable=intelligent_alerts)
    t_notify = PythonOperator(task_id="notify_team", python_callable=lambda: notify_team("Daily MLOps", "Pipeline finished. Check MLflow & data."))
    t_cleanup = PythonOperator(task_id="cleanup_old_runs", python_callable=lambda: cleanup_old_runs(keep_days=30))


    
    # linear-ish flow, branches where appropriate
    t_valid >> t_preproc >> t_anom >> t_regime >> t_train
    t_train >> t_backtest >> t_drift >> t_should
    # explanation, blending and registration are parallel after train/backtest
    t_backtest >> t_explain
    t_backtest >> t_blend
    t_backtest >> t_register
    # forecasting and dashboard after drift check
    t_should >> t_forecast >> t_dashboard >>t_alert >>  t_notify
    # cleanup can run at end
    t_notify >> t_cleanup


   