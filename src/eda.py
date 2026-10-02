import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(data_path="data/raw/heart_disease.csv", output_dir="screenshots"):
    """Generates professional EDA plots and saves them for reporting."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    
    # Binary target mapping: 0 = Absence, >0 = Presence of Heart Disease
    target_col = 'num' if 'num' in df.columns else 'target'
    if target_col in df.columns:
        df['target'] = (df[target_col] > 0).astype(int)
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # 1. Class Balance Plot
    plt.figure(figsize=(6, 4))
    sns.countplot(x='target', data=df, palette='mako')
    plt.title('Heart Disease Target Class Balance (0 = No, 1 = Yes)')
    plt.xlabel('Target (Heart Disease)')
    plt.ylabel('Patient Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_class_balance.png"), dpi=300)
    plt.close()
    
    # 2. Feature Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_correlation_heatmap.png"), dpi=300)
    plt.close()
    
    # 3. Feature Histograms
    df.hist(figsize=(12, 10), bins=15, color='teal', edgecolor='black')
    plt.suptitle('Feature Distributions', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_feature_histograms.png"), dpi=300)
    plt.close()
    
    print(f"EDA plots saved successfully to '{output_dir}/'")

if __name__ == "__main__":
    run_eda()
