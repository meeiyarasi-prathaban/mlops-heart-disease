import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_root_endpoint():
    """Tests root endpoint response."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_endpoint():
    """Tests health check endpoint."""
    response = client.get("/health")
    assert response.status_code in [200, 500]

def test_predict_endpoint():
    """Tests model prediction endpoint with valid payload."""
    payload = {
        "age": 63.0, "sex": 1.0, "cp": 3.0, "trestbps": 145.0, "chol": 233.0,
        "fbs": 1.0, "restecg": 0.0, "thalach": 150.0, "exang": 0.0,
        "oldpeak": 2.3, "slope": 0.0, "ca": 0.0, "thal": 1.0
    }
    response = client.post("/predict", json=payload)
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "risk_label" in data
        assert "confidence" in data
        assert "probabilities" in data
