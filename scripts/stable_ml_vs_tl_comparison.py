import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse
import os

# 1. SETUP & SEEDING (The "Answer to Everything")
SEED = 42
np.random.seed(SEED)

parser = argparse.ArgumentParser(description='Stable ML vs Transfer Learning Comparison')
parser.add_argument('file_2017', help='Path to 2017 dataset (Pre-train)')
parser.add_argument('file_2018', help='Path to 2018 dataset (Train/Test)')
args = parser.parse_args()

# 2. DATA LOADING
def load_data(path):
    print(f"Loading {path}...")
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna(0)
    X = df.drop(columns=['Label']).values
    y = df['Label'].astype(int).values
    return X, y

X17, y17 = load_data(args.file_2017)
X18, y18 = load_data(args.file_2018)

# Standardize based on 2017 (for TL context)
scaler = StandardScaler().fit(X17)
X17_scaled = scaler.transform(X17)
X18_scaled = scaler.transform(X18)

# 3. PRE-TRAINING THE "EXPERT" (2017 Model)
print("Pre-training Expert model on 2017 data...")
expert_2017 = xgb.XGBClassifier(n_estimators=100, random_state=SEED, eval_metric='logloss')
expert_2017.fit(X17_scaled, y17)

# 4. EXPERIMENT SETUP
percentages = np.arange(1, 11)
f1_rf = []
f1_xgb = []
f1_tl = []

# Reserve a consistent 20% of 2018 for testing
test_size = int(0.2 * len(X18))
indices = np.arange(len(X18))
np.random.shuffle(indices)
test_idx = indices[:test_size]
train_pool_idx = indices[test_size:]

X_test = X18_scaled[test_idx]
y_test = y18[test_idx]

# Pre-calculate the Expert's opinion on the test set to save time
expert_test_opinion = expert_2017.predict_proba(X_test)[:, 1].reshape(-1, 1)

print("\nStarting Sample Efficiency Loop...")

for p in percentages:
    # Calculate sample size from the training pool
    sample_size = int((p / 100) * len(X18))
    train_idx = np.random.choice(train_pool_idx, sample_size, replace=False)
    
    X_train = X18_scaled[train_idx]
    y_train = y18[train_idx]
    
    # --- Traditional ML: Random Forest ---
    rf = RandomForestClassifier(n_estimators=50, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    f1_rf.append(f1_score(y_test, rf.predict(X_test)))
    
    # --- Traditional ML: XGBoost ---
    trad_xgb = xgb.XGBClassifier(n_estimators=50, random_state=SEED, eval_metric='logloss')
    trad_xgb.fit(X_train, y_train)
    f1_xgb.append(f1_score(y_test, trad_xgb.predict(X_test)))
    
    # --- Transfer Learning: Stacking Method ---
    # 1. Get Expert Opinion on current training sample
    expert_train_opinion = expert_2017.predict_proba(X_train)[:, 1].reshape(-1, 1)
    
    # 2. Augment features with the Expert's probability
    X_train_aug = np.hstack((X_train, expert_train_opinion))
    X_test_aug = np.hstack((X_test, expert_test_opinion))
    
    # 3. Train the Stacker
    tl_stacker = xgb.XGBClassifier(n_estimators=50, random_state=SEED, eval_metric='logloss')
    tl_stacker.fit(X_train_aug, y_train)
    f1_tl.append(f1_score(y_test, tl_stacker.predict(X_test_aug)))
    
    print(f"Data: {p}% | RF: {f1_rf[-1]:.3f} | XGB: {f1_xgb[-1]:.3f} | TL: {f1_tl[-1]:.3f}")

# 5. VISUALIZATION
plt.figure(figsize=(10, 6))
plt.plot(percentages, f1_rf, label='Traditional Random Forest', marker='o', color='#95a5a6', linestyle='--')
plt.plot(percentages, f1_xgb, label='Traditional XGBoost', marker='s', color='#27ae60', linestyle='--')
plt.plot(percentages, f1_tl, label='TL Stacking (2017 Expert + 2018 Learning)', marker='^', color='#2980b9', linewidth=2.5)

plt.title("Model Resilience & Sample Efficiency (1% - 10% Data)", fontsize=14)
plt.xlabel("Percentage of 2018 Data Used for Training (%)", fontsize=12)
plt.ylabel("F1-Score (Test Set)", fontsize=12)
plt.xticks(percentages)
plt.ylim(0, 1.0) # Set Y-axis to full range for perspective
plt.grid(alpha=0.3, linestyle=':')
plt.legend(loc='lower right')
plt.tight_layout()

plt.savefig('stable_ml_vs_tl_comparison.png', dpi=300)
print("\nSuccess! Comparison graph saved as 'stable_ml_vs_tl_comparison.png'")
plt.show()