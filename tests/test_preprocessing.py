import os
import pandas as pd
import pytest
from src.preprocessing import preprocess_data

def test_preprocess_data_execution(tmp_path):
    """Verifies that preprocess_data correctly generates output datasets and models."""
    dummy_df = pd.DataFrame({
        'age': [63, 67, 67, 37, 41],
        'sex': [1, 1, 1, 1, 0],
        'cp': [1, 4, 4, 3, 2],
        'trestbps': [145, 160, 120, 130, 130],
        'chol': [233, 286, 229, 250, 204],
        'fbs': [1, 0, 0, 0, 0],
        'restecg': [2, 2, 2, 0, 2],
        'thalach': [150, 108, 129, 187, 172],
        'exang': [0, 1, 1, 0, 0],
        'oldpeak': [2.3, 1.5, 2.6, 3.5, 1.4],
        'slope': [3, 2, 2, 3, 1],
        'ca': [0, 3, 2, 0, 0],
        'thal': [6, 3, 7, 3, 3],
        'num': [0, 2, 1, 0, 0]
    })
    
    raw_path = tmp_path / "raw_heart.csv"
    out_dir = tmp_path / "processed"
    dummy_df.to_csv(raw_path, index=False)
    
    preprocess_data(raw_data_path=str(raw_path), output_dir=str(out_dir))
    
    assert os.path.exists(out_dir / "train.csv")
    assert os.path.exists(out_dir / "test.csv")
    assert os.path.exists(out_dir / "scaler.pkl")
    assert os.path.exists(out_dir / "imputer.pkl")
