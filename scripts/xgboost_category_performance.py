import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse
import os

# 1. SETUP COMMAND LINE ARGUMENTS
parser = argparse.ArgumentParser(description='XGBoost IDS Categorized Transfer Learning')
parser.add_argument('train_file', help='Path to 2017 dataset (CSV)')
parser.add_argument('test_file', help='Path to 2018 dataset (CSV)')
args = parser.parse_args()

# 2. UNIFIED BEHAVIORAL MAPPING
attack_map = {
    'BENIGN': 'BENIGN',
    'DDoS': 'Volumetric', 'DoS Hulk': 'Volumetric', 'DoS GoldenEye': 'Volumetric',
    'DoS Slowhttptest': 'Volumetric', 'DoS slowloris': 'Volumetric',
    'FTP-Patator': 'Authentication', 'SSH-Patator': 'Authentication', 'Web Attack - Brute Force': 'Authentication',
    'Web Attack - Sql Injection': 'Web/Injection', 'Web Attack - XSS': 'Web/Injection',
    'PortScan': 'Reconnaissance',
    'Bot': 'Persistence', 'Infiltration': 'Persistence'
}

# 3. DATA LOADING AND CLEANING
def load_and_clean(path):
    print(f"Reading {path}...")
    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df['Category'] = df['Label'].map(attack_map).fillna('BENIGN')
    df['BinaryLabel'] = (df['Category'] != 'BENIGN').astype(int)
    return df

print("Loading datasets...")
df_train = load_and_clean(args.train_file)
df_test = load_and_clean(args.test_file)

# 4. PREPARE DATA
X_train_raw = df_train.drop(columns=['Label', 'Category', 'BinaryLabel']).values
y_train = df_train['BinaryLabel'].values

# Scale features (XGBoost is less sensitive than MLP, but it's good for consistency)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(df_test.drop(columns=['Label', 'Category', 'BinaryLabel']).values)

# 5. TRAIN XGBOOST
print("Training XGBoost on 2017 data...")
# use 'binary:logistic' to get a probability output (0-1)
clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='binary:logistic',
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
clf.fit(X_train_scaled, y_train)

# 6. CATEGORIZED EVALUATION ON 2018
categories = [c for c in df_test['Category'].unique() if c != 'BENIGN']
category_f1s = {}

print("\nEvaluating XGBoost by category on 2018 data...")
for cat in categories:
    mask = (df_test['Category'] == 'BENIGN') | (df_test['Category'] == cat)
    X_cat = X_test_scaled[mask]
    y_cat = df_test['BinaryLabel'].values[mask]

    # Get probabilities (needed for the adaptive threshold)
    probs = clf.predict_proba(X_cat)[:, 1]

    # Adaptive Thresholding (Window=50, 95th Percentile)
    preds = []
    window_size = 50
    for i in range(len(probs)):
        win = probs[max(0, i-window_size):i+1]
        thresh = np.quantile(win, 0.95)
        preds.append(1 if probs[i] >= thresh else 0)

    score = f1_score(y_cat, preds)
    category_f1s[cat] = score
    print(f"Category: {cat:<15} | F1: {score:.4f}")

# 7. VISUALIZATION
plt.figure(figsize=(10, 6))
bars = plt.bar(category_f1s.keys(), category_f1s.values(), color='#27ae60')
plt.title("XGBoost + Adaptive Threshold: F1-Score by Category (2018)", fontsize=14)
plt.ylabel("F1-Score")
plt.xlabel("Attack Category")

# Dynamic Y-axis Scaling
max_f1 = max(category_f1s.values()) if category_f1s else 1.0
plt.ylim(0, max_f1 + 0.1)

# Add value labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 3), ha='center')

plt.tight_layout()
plt.savefig('xgboost_category_performance.png', dpi=300)
print("\nGraph saved as 'xgboost_category_performance.png'")
plt.show()