import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier

def load_clean(path):
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna(0)
    drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df['Label'] = df['Label'].astype(str).str.lower().str.strip()
    return df

df17 = load_clean('cic2017_training_av_lbl_100K_hdr.csv')
df18 = load_clean('cic2018_training_av_lbl_100K_hdr.csv')
features = [f for f in df17.columns if f != 'Label']

# Determine Importance
X17 = df17[features].values
y17 = (df17['Label'] != 'benign').astype(int).values
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X17, y17)

# Calculate Drift for Top 5
importances = rf.feature_importances_
top_5_idx = np.argsort(importances)[-5:][::-1]

print(f"{'Feature':<30} | {'Importance':<10} | {'K-S Statistic (D)':<15}")
print("-" * 65)

for i in top_5_idx:
    col = features[i]
    # Calculate K-S Statistic
    d_stat, p_val = ks_2samp(df17[col], df18[col])
    print(f"{col:<30} | {importances[i]:<10.4f} | {d_stat:<15.4f}")