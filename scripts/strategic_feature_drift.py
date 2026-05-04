import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

def load_clean(path):
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna(0)
    drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df['Label'] = df['Label'].astype(str).str.strip()
    return df

print("Loading data and determining feature importance...")
df17 = load_clean('cic2017_training_av_lbl_100K_hdr.csv')
df18 = load_clean('cic2018_training_av_lbl_100K_hdr.csv')
features = [f for f in df17.columns if f != 'Label']

# Train a temporary model to find what matters most in 2017
X17 = df17[features].values
y17 = (df17['Label'].str.lower() != 'benign').astype(int).values
temp_model = xgb.XGBClassifier().fit(X17, y17)

# Get Top 5 Features
importances = temp_model.feature_importances_
top_5_idx = np.argsort(importances)[-5:][::-1]
top_features = [features[i] for i in top_5_idx]

print(f"Top 5 Critical Features: {top_features}")

# Visualizing Drift for these specific features
plt.figure(figsize=(15, 10))
for i, col in enumerate(top_features):
    plt.subplot(2, 3, i+1)
    # Use log-scale for network data as it's often highly skewed
    sns.kdeplot(df17[col], label='2017 (Source)', color='#2980b9', fill=True, alpha=0.3)
    sns.kdeplot(df18[col], label='2018 (Target)', color='#c0392b', fill=True, alpha=0.3)
    plt.title(f'Drift in Key Feature:\n{col}', fontsize=12)
    plt.xlabel('Scaled Value')
    plt.ylabel('Density')
    plt.legend()

plt.tight_layout()
plt.savefig('strategic_feature_drift.png', dpi=300)
plt.show()