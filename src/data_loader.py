import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

def load_heart_disease_data(output_path="data/raw/heart_disease.csv"):
    """Fetches the UCI Heart Disease dataset and saves it as a CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("Fetching dataset from UCI ML Repository...")
    
    # UCI ID for Heart Disease dataset is 45
    heart_disease = fetch_ucirepo(id=45)
    
    X = heart_disease.data.features
    y = heart_disease.data.targets
    
    df = pd.concat([X, y], axis=1)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully saved to {output_path} (Shape: {df.shape})")
    return df

if __name__ == "__main__":
    load_heart_disease_data()
