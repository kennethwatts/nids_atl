import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. LOAD & PREP
df = pd.read_csv("cic2018_training_av_lbl_100K_hdr.csv")
df['Label'] = df['Label'].astype(str).str.strip()

# 2. GROUPING INTO FAMILIES (Crucial for small samples)
# This aggregates small samples into a larger, statistically significant group
family_map = {
    'DoS Hulk': 'DoS', 'DoS Slowhttptest': 'DoS', 'DoS slowloris': 'DoS', 'DoS GoldenEye': 'DoS',
    'DDoS': 'DDoS', 
    'FTP-Patator': 'BruteForce', 'SSH-Bruteforce': 'BruteForce',
    'Web Attack - Brute Force': 'Web', 'Web Attack - XSS': 'Web'
}
df['Family'] = df['Label'].map(family_map).fillna(df['Label'])

# 3. ANALYSIS
X = df.drop(columns=['Label', 'Family']).replace([np.inf, -np.inf], np.nan).fillna(0)
y = df['Family']

for family in [f for f in y.unique() if f != 'Benign']:
    y_bin = (y == family).astype(int)
    
    # Check if we have enough to train
    if y_bin.sum() < 20:
        print(f"Skipping {family}: Only {y_bin.sum()} samples found.")
        continue

    # 'balanced_subsample' forces the trees to care about the minority class
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', random_state=42)
    rf.fit(X, y_bin)
    
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(f"\n--- {family} Signature (Top 3) ---")
    print(importances.head(3))