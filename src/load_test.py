import time
import random
import requests

API_URL = "http://localhost:8000/predict"

sample_patients = [
    {"age": 63.0, "sex": 1.0, "cp": 3.0, "trestbps": 145.0, "chol": 233.0, "fbs": 1.0, "restecg": 0.0, "thalach": 150.0, "exang": 0.0, "oldpeak": 2.3, "slope": 0.0, "ca": 0.0, "thal": 1.0},
    {"age": 37.0, "sex": 1.0, "cp": 2.0, "trestbps": 130.0, "chol": 250.0, "fbs": 0.0, "restecg": 1.0, "thalach": 187.0, "exang": 0.0, "oldpeak": 3.5, "slope": 0.0, "ca": 0.0, "thal": 2.0},
    {"age": 41.0, "sex": 0.0, "cp": 1.0, "trestbps": 130.0, "chol": 204.0, "fbs": 0.0, "restecg": 0.0, "thalach": 172.0, "exang": 0.0, "oldpeak": 1.4, "slope": 1.0, "ca": 0.0, "thal": 2.0},
    {"age": 56.0, "sex": 1.0, "cp": 3.0, "trestbps": 120.0, "chol": 236.0, "fbs": 0.0, "restecg": 1.0, "thalach": 178.0, "exang": 0.0, "oldpeak": 0.8, "slope": 1.0, "ca": 0.0, "thal": 2.0}
]

print("Starting automated load generator... Press Ctrl+C to stop.")
successful_requests = 0

try:
    for i in range(1, 101):
        patient = random.choice(sample_patients)
        try:
            response = requests.post(API_URL, json=patient, timeout=2)
            if response.status_code == 200:
                successful_requests += 1
                res_json = response.json()
                print(f"[{i}/100] Success -> Prediction: {res_json['risk_label']} (Confidence: {res_json['confidence']})")
            else:
                print(f"[{i}/100] Failed -> Status Code: {response.status_code}")
        except requests.exceptions.RequestException:
            print(f"[{i}/100] Connection Error: Ensure server is running on http://localhost:8000")
            time.sleep(1)
            continue
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nLoad test interrupted by user.")

print(f"\nCompleted! Total successful requests sent: {successful_requests}/100")
