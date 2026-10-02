# End-to-End MLOps Pipeline for Heart Disease Risk Prediction

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.0+-0194E2.svg)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Orchestrated-326CE5.svg)
![CI/CD](https://img.shields.io/badge/GitHub_Actions-Automated_CI%2FCD-2088FF.svg)

Production-ready, fully automated end-to-end Machine Learning Operations (MLOps) pipeline built on the **UCI Heart Disease Dataset**. Features automated data ingestion, preprocessing, MLflow experiment tracking, FastAPI inference serving, Docker containerization, Kubernetes orchestration, Prometheus observability, and GitHub Actions CI/CD.

---

## Technical Architecture

```
[ UCI Repository ] ---> [ Data Loader ] ---> [ Preprocessing & Scaling ]
                                                        |
                                                        v
[ Prometheus UI ] <--- [ FastAPI Server ] <--- [ MLflow Model Registry ]
      ^                      |                          |
      |                      v                          v
[ Metrics Port ]      [ Docker / K8s ]          [ Logistic Regression ]
                      [ Deployment   ]          [ Random Forest       ]
```

---

## System Tech Stack

| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.12 |
| **Data & Modeling** | Pandas, Scikit-Learn, Joblib, Skops |
| **Experiment Tracking** | MLflow |
| **API Framework** | FastAPI, Uvicorn, Pydantic |
| **Containerization** | Docker |
| **Orchestration** | Kubernetes (Deployment, Service, ConfigMap) |
| **Monitoring** | Prometheus, Prometheus FastAPI Instrumentator |
| **Testing & Quality** | Pytest, Flake8 |
| **CI/CD Automation** | GitHub Actions |

---

## Directory Structure

```
mlops-heart-disease/
├── .github/
│   └── workflows/
│       └── ci_cd.yml          # GitHub Actions CI/CD Pipeline
├── data/
│   ├── raw/                   # Raw UCI Heart Disease CSV
│   └── processed/             # Scaled train/test sets and fitted pkl artifacts
├── k8s/
│   ├── deployment.yaml        # K8s Deployment Manifest
│   ├── service.yaml           # K8s Service Manifest (NodePort 30080)
│   └── monitoring.yaml        # K8s Prometheus Monitoring Manifest
├── models/
│   └── best_model.pkl         # Production model artifact
├── screenshots/               # EDA plots, confusion matrices, ROC curves
├── src/
│   ├── __init__.py
│   ├── app.py                 # FastAPI REST application
│   ├── data_loader.py         # Data ingestion script
│   ├── eda.py                 # EDA plotting module
│   ├── load_test.py           # Traffic generator for Prometheus metrics
│   ├── preprocessing.py       # Cleaning, scaling, and splitting pipeline
│   └── train.py               # Model training & MLflow tracking script
├── tests/
│   ├── test_api.py            # API integration tests
│   └── test_preprocessing.py  # Pipeline unit tests
├── Dockerfile                 # Container image specification
├── pytest.ini                 # Test environment configuration
└── requirements.txt           # Project dependencies
```

---

## Quick Start Guide

### 1. Repository Setup & Environment
```bash
git clone [https://github.com/meeiyarasi-prathaban/mlops-heart-disease.git](https://github.com/meeiyarasi-prathaban/mlops-heart-disease.git)
cd mlops-heart-disease
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

### 2. Data Pipeline & Model Training
```bash
python src/data_loader.py
python src/eda.py
python src/preprocessing.py
python src/train.py
```

### 3. Interactive MLflow UI
```bash
mlflow ui
```
View experiments at `http://127.0.0.1:5000`.

### 4. Run API Locally
```bash
uvicorn src.app:app --port 8000
```
- Interactive Swagger API Docs: `http://127.0.0.1:8000/docs`
- Prometheus Metrics Route: `http://127.0.0.1:8000/metrics`

### 5. Automated Tests
```bash
python -m pytest -v
```

### 6. Docker Containerization
```bash
docker build -t heart-disease-api:v1 .
docker run -d -p 8000:8000 --name heart-disease-app heart-disease-api:v1
```

### 7. Kubernetes Deployment & Monitoring
```bash
kubectl apply -f k8s/
kubectl get pods,svc
```

### 8. Load Generator
```bash
python src/load_test.py
```

---

## API Documentation

### POST `/predict`
**Sample Request Body:**
```json
{
  "age": 63.0,
  "sex": 1.0,
  "cp": 3.0,
  "trestbps": 145.0,
  "chol": 233.0,
  "fbs": 1.0,
  "restecg": 0.0,
  "thalach": 150.0,
  "exang": 0.0,
  "oldpeak": 2.3,
  "slope": 0.0,
  "ca": 0.0,
  "thal": 1.0
}
```

**Sample Response:**
```json
{
  "prediction": 1,
  "risk_label": "High Risk",
  "confidence": 0.8842,
  "probabilities": {
    "low_risk": 0.1158,
    "high_risk": 0.8842
  }
}
```
