# End-to-End Cryptocurrency Forecasting & Monitoring


## **Description**

**CryptoMLOps** is a fully automated **MLOps pipeline** for cryptocurrency time series forecasting. The project combines:

* **Data ingestion** from CSV or API
* **Data preprocessing & feature engineering**
* **Anomaly detection** and **volatility regime detection**
* **ARIMA + GARCH model training**
* **Backtesting & drift detection**
* **Forecasting with uncertainty intervals**
* **Model monitoring & dashboard generation**
* **MLflow logging & experiment tracking**
* **Automated notification & cleanup**

The pipeline is designed for **reproducibility**, **monitoring**, and **continuous learning**. Built with **Python**, **Airflow**, **MLflow**, and **Docker**, it is ready for production deployment.

---

## **Features**

* Fetch, preprocess, and validate cryptocurrency datasets
* Detect anomalies and regime changes in market behavior
* Train ARIMA + GARCH models and log metrics & artifacts to MLflow
* Backtest models using walk-forward validation
* Detect dataset drift using PSI and KS statistics
* Generate forecasts with uncertainty intervals
* Blend multiple models for ensemble predictions
* Explain predictions using SHAP for tree models
* Monitor model performance and alert on issues
* Automated cleanup of old runs and artifacts
* Fully orchestrated via **Apache Airflow DAGs**

---

## **Architecture**

```
┌──────────────┐
│ Fetch Data   │
└─────┬────────┘
      │
┌─────▼────────┐
│ Preprocessing│
└─────┬────────┘
      │
┌─────▼──────────┐
│ Anomaly Detection│
└─────┬──────────┘
      │
┌─────▼──────────┐
│ Regime Detection│
└─────┬──────────┘
      │
┌─────▼──────────┐
│ Train ARIMA+GARCH│
└─────┬──────────┘
      │
┌─────▼──────────┐
│ Backtesting      │
└─────┬──────────┘
      │
┌─────▼──────────┐
│ Drift Detection │
└─────┬──────────┘
      │
┌─────▼──────────┐
│ Forecast & Dashboard│
└─────┬──────────┘
      │
┌─────▼──────────┐
│ Notifications & Cleanup │
└───────────────────────┘
```

---

## **Tech Stack**

* **Python** 3.11
* **Apache Airflow** 2.8.1 (DAG orchestration)
* **MLflow** (experiment tracking & model registry)
* **Docker & Docker Compose** (containerization)
* **Pandas / NumPy / Scikit-learn** (data processing & ML)
* **Statsmodels / Arch** (ARIMA + GARCH)
* **Joblib** (model serialization)
* **Matplotlib / Seaborn** (visualization)
* **SHAP** (feature importance)
* Optional: SMTP for notifications

---

## **Installation**

1. Clone the repository:

```bash
git clone https://github.com/gamzeakkurt/cryptomlops.git
cd cryptomlops
```

2. Build and start containers:

```bash
docker-compose build
docker-compose up
```

3. Access services:

* **Airflow:** `http://localhost:8081`
* **MLflow:** `http://localhost:5050`
* **FastAPI app (if deployed):** `http://localhost:8001`

---

## **Directory Structure**

```
.
├── dags/                     # Airflow DAGs
│   └── ml_pipeline_full.py
├── src/                      # Python modules for each task
│   ├── data_pipeline.py
│   ├── train.py
│   ├── backtest.py
│   ├── drift_detector.py
│   ├── detect_anomalies.py
│   ├── detect_regime.py
│   ├── forecast_with_uncertainty.py
│   ├── explainability.py
│   ├── should_retrain.py
│   ├── register_model.py
│   ├── generate_dashboard.py
│   ├── notify_team.py
│   └── cleanup_old_runs.py
├── data/                     # Raw and processed data
├── models/                   # Saved models
├── mlruns/                   # MLflow experiment logs
├── dockerfile.airflow        # Airflow Dockerfile
├── Dockerfile                # API / main app Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## **Usage**

1. Place your historical crypto data in `./data/` (CSV format)
2. Start Docker Compose as above
3. Airflow DAG `mlops_crypto_full_pipeline` will automatically execute:

   * Fetch → preprocess → anomaly → regime → train → backtest → drift → forecast → dashboard → notify
4. Check **MLflow UI** for metrics and artifacts
5. Check **dashboard** in `./data/dashboard.png`
6. Models saved in `./models`

---

## **Hyperparameter Tuning & Experimentation**

* Change ARIMA/GARCH parameters directly in `train.py`
* MLflow allows logging multiple experiments & parameters

---

## **Contributing**

* Fork the repository
* Create a feature branch: `git checkout -b feature/my-feature`
* Commit changes: `git commit -am 'Add new feature'`
* Push to branch: `git push origin feature/my-feature`
* Open a pull request

---

## **License**

MIT License – See `LICENSE`


