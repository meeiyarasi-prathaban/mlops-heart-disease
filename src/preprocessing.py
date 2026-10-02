import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

def preprocess_data(raw_data_path="data/raw/heart_disease.csv", output_dir="data/processed"):
    """Cleans data, scales features, splits dataset, and saves transformers."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(raw_data_path)
    
    # Standardize target column (UCI: 0 = Healthy, 1-4 = Disease)
    target_col = 'num' if 'num' in df.columns else 'target'
    if target_col in df.columns:
        df['target'] = (df[target_col] > 0).astype(int)
        if target_col != 'target':
            df.drop(columns=[target_col], inplace=True)
            
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Impute missing values using median strategy
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Train/Test Split (80/20 Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Fit scaler on training set only
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    # Save processed CSV files
    train_df = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test_scaled, y_test.reset_index(drop=True)], axis=1)
    
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    # Save transformers for inference pipeline
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    joblib.dump(imputer, os.path.join(output_dir, "imputer.pkl"))
    
    print(f"Preprocessing complete. Processed datasets and artifacts saved to '{output_dir}/'")

if __name__ == "__main__":
    preprocess_data()
