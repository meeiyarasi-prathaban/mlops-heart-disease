import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="API for predicting heart disease risk using trained ML models.",
    version="1.0.0"
)

# Initialize Prometheus Instrumentator
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "data/processed/scaler.pkl"
IMPUTER_PATH = "data/processed/imputer.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)
except Exception as e:
    model, scaler, imputer = None, None, None
    print(f"Warning: Artifacts could not be loaded: {e}")

class PatientData(BaseModel):
    age: float = Field(..., example=63.0)
    sex: float = Field(..., example=1.0)
    cp: float = Field(..., example=3.0)
    trestbps: float = Field(..., example=145.0)
    chol: float = Field(..., example=233.0)
    fbs: float = Field(..., example=1.0)
    restecg: float = Field(..., example=0.0)
    thalach: float = Field(..., example=150.0)
    exang: float = Field(..., example=0.0)
    oldpeak: float = Field(..., example=2.3)
    slope: float = Field(..., example=0.0)
    ca: float = Field(..., example=0.0)
    thal: float = Field(..., example=1.0)

@app.get("/")
def read_root():
    return {"message": "Heart Disease Prediction API is active. Access /docs for Swagger UI or /metrics for Prometheus metrics."}

@app.get("/health")
def health_check():
    if model is None or scaler is None or imputer is None:
        raise HTTPException(status_code=500, detail="Model or preprocessors not loaded.")
    return {"status": "healthy", "model_type": type(model).__name__}

@app.post("/predict")
def predict(data: PatientData):
    if model is None or scaler is None or imputer is None:
        raise HTTPException(status_code=500, detail="Model or preprocessors not loaded.")
    
    try:
        input_dict = data.model_dump() if hasattr(data, 'model_dump') else data.dict()
        df = pd.DataFrame([input_dict])
        
        imputed_data = imputer.transform(df)
        scaled_data = scaler.transform(imputed_data)
        
        prediction = int(model.predict(scaled_data)[0])
        probabilities = model.predict_proba(scaled_data)[0]
        confidence = float(probabilities[prediction])
        
        return {
            "prediction": prediction,
            "risk_label": "High Risk" if prediction == 1 else "Low Risk",
            "confidence": round(confidence, 4),
            "probabilities": {
                "low_risk": round(float(probabilities[0]), 4),
                "high_risk": round(float(probabilities[1]), 4)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")
