from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
from joblib import load
import os
import sklearn

app = FastAPI()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = None

# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    features: Dict[str, float]

class BatchPredictionRequest(BaseModel):
    features_list: List[Dict[str, float]]

class PredictionResponse(BaseModel):
    prediction: float
    input_features: Dict[str, float]
    status: str = "success"

class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    input_features: List[Dict[str, float]]
    count: int
    status: str = "success"

@app.on_event("startup")
def load_model():
    global model
    try:
        print(f"🔍 scikit-learn version: {sklearn.__version__}")
        print(f"🔍 Loading model from: {MODEL_PATH}")
        model = load(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"⚠️ Could not load model: {e}")
        model = None


@app.post("/predict", response_model=PredictionResponse)
def predict(features: Dict[str, float]):
    """
    Predict a single value based on input features.
    
    Accepts features as a dictionary: {"return_lag_1": 0.01, "return_lag_2": 0.02, ...}
    
    Returns the predicted value along with the input features for verification.
    """
    if model is None:
        return {
            "prediction": 0.0,
            "input_features": features,
            "status": "error: Model not loaded"
        }
    
    try:
        # Convert features to DataFrame
        df = pd.DataFrame([features])
        
        # Ensure features are in the correct order (if model has feature names)
        if hasattr(model, 'feature_names_in_'):
            # Reorder columns to match training data
            df = df.reindex(columns=model.feature_names_in_, fill_value=0)
        
        # Make prediction
        pred = model.predict(df)[0]
        
        return {
            "prediction": float(pred),
            "input_features": features,
            "status": "success"
        }
    except Exception as e:
        return {
            "prediction": 0.0,
            "input_features": features,
            "status": f"error: {str(e)}"
        }


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    """
    Predict multiple values at once based on a list of input features.
    Returns all predicted values along with their corresponding input features.
    """
    if model is None:
        return {
            "predictions": [],
            "input_features": request.features_list,
            "count": 0,
            "status": "error: Model not loaded"
        }
    
    try:
        # Convert list of features to DataFrame
        df = pd.DataFrame(request.features_list)
        
        # Ensure features are in the correct order (if model has feature names)
        if hasattr(model, 'feature_names_in_'):
            # Reorder columns to match training data
            df = df.reindex(columns=model.feature_names_in_, fill_value=0)
        
        # Make predictions
        predictions = model.predict(df)
        
        return {
            "predictions": [float(pred) for pred in predictions],
            "input_features": request.features_list,
            "count": len(predictions),
            "status": "success"
        }
    except Exception as e:
        return {
            "predictions": [],
            "input_features": request.features_list,
            "count": 0,
            "status": f"error: {str(e)}"
        }


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "Crypto Analysis Prediction API",
        "endpoints": {
            "/predict": "POST - Predict a single value",
            "/predict/batch": "POST - Predict multiple values",
            "/health": "GET - Check API health"
        }
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }
