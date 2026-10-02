import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, RocCurveDisplay

def train_and_evaluate():
    """Trains Logistic Regression and Random Forest models with MLflow tracking."""
    # Ensure model output directory exists
    os.makedirs("models", exist_ok=True)
    os.makedirs("screenshots", exist_ok=True)
    
    # Load preprocessed datasets
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']
    
    # Set MLflow experiment name
    mlflow.set_experiment("Heart_Disease_Prediction")
    
    models = {
        "Logistic_Regression": LogisticRegression(random_state=42, max_iter=1000, C=1.0),
        "Random_Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5)
    }
    
    best_roc_auc = 0.0
    best_model = None
    best_model_name = ""

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            print(f"Training {model_name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            cv_mean = np.mean(cv_scores)
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Predict on test set
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
            
            # Compute evaluation metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)
            
            print(f"{model_name} -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, ROC-AUC: {roc_auc:.4f}")
            
            # Log Parameters
            if model_name == "Logistic_Regression":
                mlflow.log_param("C", 1.0)
                mlflow.log_param("max_iter", 1000)
            elif model_name == "Random_Forest":
                mlflow.log_param("n_estimators", 100)
                mlflow.log_param("max_depth", 5)
                
            # Log Metrics
            mlflow.log_metric("cv_roc_auc", cv_mean)
            mlflow.log_metric("test_accuracy", acc)
            mlflow.log_metric("test_precision", prec)
            mlflow.log_metric("test_recall", rec)
            mlflow.log_metric("test_roc_auc", roc_auc)
            
            # Save and log Confusion Matrix plot artifact
            plt.figure(figsize=(5, 4))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            cm_path = f"screenshots/cm_{model_name}.png"
            plt.tight_layout()
            plt.savefig(cm_path)
            plt.close()
            mlflow.log_artifact(cm_path)
            
            # Save and log ROC Curve plot artifact
            plt.figure(figsize=(5, 4))
            RocCurveDisplay.from_estimator(model, X_test, y_test)
            plt.title(f'ROC Curve - {model_name}')
            roc_path = f"screenshots/roc_{model_name}.png"
            plt.tight_layout()
            plt.savefig(roc_path)
            plt.close()
            mlflow.log_artifact(roc_path)
            
            # Log Model Artifact with skops_trusted_types
            mlflow.sklearn.log_model(
                model, 
                artifact_path="model",
                skops_trusted_types=["sklearn.tree._tree.Tree"]
            )
            
            # Keep track of best performing model
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_model = model
                best_model_name = model_name

    # Save best model to disk for API serving
    best_model_path = "models/best_model.pkl"
    joblib.dump(best_model, best_model_path)
    print(f"\nBest model: {best_model_name} with ROC-AUC = {best_roc_auc:.4f} saved to '{best_model_path}'")

if __name__ == "__main__":
    train_and_evaluate()
