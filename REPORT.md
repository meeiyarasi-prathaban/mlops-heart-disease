# End-to-End MLOps Pipeline for Heart Disease Risk Prediction
**Technical Assignment Report**

---

## Executive Summary

This report documents the architectural design, implementation, and evaluation of an end-to-end Machine Learning Operations (MLOps) pipeline developed for predicting heart disease risk using the **UCI Heart Disease Dataset**. The primary objective is to transition from static model training to an automated, resilient, and observable production workflow.

The operational pipeline encompasses automated ingestion, pre-processing, experiment tracking using **MLflow**, serving via a **FastAPI** REST interface, containerization through **Docker**, cluster orchestration via **Kubernetes**, real-time metric tracking using **Prometheus**, and continuous integration/deployment via **GitHub Actions**.

---

## 1. Dataset & Exploratory Data Analysis (EDA)

### 1.1 Dataset Attributes
The system utilizes the classic UCI Heart Disease Dataset (ID: 45) comprising 14 core physiological attributes:
- **Demographics:** `age`, `sex`
- **Clinical Measurements:** `cp` (chest pain type), `trestbps` (resting blood pressure), `chol` (serum cholesterol), `fbs` (fasting blood sugar), `restecg` (resting ECG), `thalach` (maximum heart rate achieved), `exang` (exercise-induced angina), `oldpeak` (ST depression), `slope` (ST slope), `ca` (colored vessels), `thal` (thalassemia).
- **Target (`target`):** Binary conversion where `0` indicates no diagnosis of heart disease and `1` indicates presence of heart disease.

### 1.2 Automated Data Processing
- Missing values within numeric features are imputed using median values (`SimpleImputer`).
- Features are normalized using `StandardScaler` fitted strictly on the training partition to prevent data leakage.
- Processed artifacts (`scaler.pkl`, `imputer.pkl`) are serialized for real-time inference preprocessing.

---

## 2. Model Training & MLflow Experiment Tracking

Two candidate classifiers were evaluated: **Logistic Regression** (baseline) and **Random Forest Classifier** (ensemble model).

### 2.1 Experiment Setup & Logging
Experiments were tracked via MLflow. Metrics logged include Cross-Validation ROC-AUC, Test Accuracy, Precision, Recall, and Test ROC-AUC. Visual artifacts (Confusion Matrix and ROC Curve plots) were generated and logged automatically.

### 2.2 Model Performance Summary

| Model | CV ROC-AUC | Test Accuracy | Precision | Recall | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8920 | 0.8689 | 0.8125 | 0.9286 | 0.9513 |
| **Random Forest** | 0.9105 | 0.8852 | 0.8387 | 0.9286 | **0.9589** |

**Selection:** The **Random Forest Classifier** achieved superior ROC-AUC (0.9589) and overall accuracy (88.52%), and was registered as the primary inference engine (`models/best_model.pkl`).

---

## 3. Serving & Containerization

### 3.1 REST API Development (FastAPI)
The service layer exposes three core endpoints:
1. `GET /`: Health status and landing documentation link.
2. `GET /health`: Model availability and preprocessor status.
3. `POST /predict`: Real-time vector inference returning classification, risk label, confidence score, and probability distribution.
4. `GET /metrics`: Prometheus exporter endpoint generating real-time metric counters.

### 3.2 Docker Containerization
A lightweight container based on `python:3.10-slim` packages the dependencies, API application code, and pre-trained binary model files, ensuring reproducible runtime environments.

---

## 4. Kubernetes Orchestration & Observability

### 4.1 Kubernetes Architecture
- **Deployment (`k8s/deployment.yaml`):** Runs 2 stateless container replicas with defined CPU/memory request boundaries and automated liveness/readiness health probes attached to `/health`.
- **Service (`k8s/service.yaml`):** Exposes pods across the cluster on NodePort `30080`.
- **Monitoring (`k8s/monitoring.yaml`):** Deploys a dedicated Prometheus instance scraping `/metrics` every 5 seconds.

### 4.2 System Metrics Tracked
- Request throughput (`http_requests_total`)
- Request latency percentiles (`http_request_duration_seconds`)
- System error frequencies and status code distributions

---

## 5. Continuous Integration & Deployment (CI/CD)

The GitHub Actions workflow (`.github/workflows/ci_cd.yml`) automates verification on every push to `main`:
1. **Linting:** Code formatting and syntax checks via `flake8`.
2. **Data Ingestion & Preprocessing Verification:** Automated dataset download and feature transformation execution.
3. **Model Retraining Test:** Verification of model fitting and artifact creation.
4. **Unit Test Execution:** Execution of `pytest` test suites (`tests/test_api.py`, `tests/test_preprocessing.py`).

---

## Conclusion

The implemented MLOps pipeline demonstrates a production-grade infrastructure for automated heart disease prediction. By combining MLflow tracking, FastAPI serving, Docker containerization, Kubernetes orchestration, and Prometheus observability, the system achieves reproducibility, scalability, and automated monitoring.
