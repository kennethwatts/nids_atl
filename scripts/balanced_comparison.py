import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse

# 1. SETUP
SEED = 42
np.random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('file_2017')
parser.add_argument('file_2018')
args = parser.parse_args()

# 2. DATA LOADING
def load_and_binarize(path):
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna(0)
    y = (df['Label'].astype(str).str.lower().str.strip() != 'benign').astype(int).values
    X = df.drop(columns=['Label']).values
    return X, y

X17, y17 = load_and_binarize(args.file_2017)
X18, y18 = load_and_binarize(args.file_2018)

scaler = StandardScaler().fit(X17)
X17_scaled = scaler.transform(X17)
X18_scaled = scaler.transform(X18)

# 3. PRE-TRAIN EXPERT
expert_2017 = xgb.XGBClassifier(n_estimators=100, random_state=SEED)
expert_2017.fit(X17_scaled, y17)

# 4. BALANCED MICRO-LOOP
# Sample Sizes: 10, 20, ... 100
sample_sizes = np.arange(10, 110, 10)
f1_rf, f1_tl = [], []

test_idx = np.random.choice(len(X18), int(0.2 * len(X18)), replace=False)
X_test, y_test = X18_scaled[test_idx], y18[test_idx]

# Indices for balancing
benign_idx = np.where(y18 == 0)[0]
attack_idx = np.where(y18 == 1)[0]

print("\nRunning Balanced Comparison...")
for s in sample_sizes:
    # Force 50/50 Balance in training
    n_each = s // 2
    s_benign = np.random.choice(benign_idx, n_each, replace=False)
    s_attack = np.random.choice(attack_idx, n_each, replace=False)
    train_idx = np.concatenate([s_benign, s_attack])
    
    X_train, y_train = X18_scaled[train_idx], y18[train_idx]
    
    # --- Supervised RF ---
    rf = RandomForestClassifier(n_estimators=50, random_state=SEED)
    rf.fit(X_train, y_train)
    f1_rf.append(f1_score(y_test, rf.predict(X_test)))

    # --- Unsupervised TL (Expert + Dynamic Quantile) ---
    probs = expert_2017.predict_proba(X_test)[:, 1]
    # We set the quantile to match the actual expected attack rate (e.g. 0.1)
    # This makes the comparison fair
    thresh = np.quantile(probs, 0.90) 
    tl_preds = (probs >= thresh).astype(int)
    f1_tl.append(f1_score(y_test, tl_preds))
    
    print(f"Size: {s:<3} | RF: {f1_rf[-1]:.3f} | TL: {f1_tl[-1]:.3f}")

# 5. VISUALIZATION
plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, f1_rf, label='Supervised RF (Balanced Samples)', marker='o', color='#e74c3c')
plt.plot(sample_sizes, f1_tl, label='Unsupervised TL (Expert Knowledge)', marker='^', color='#2980b9', linewidth=2.5)
plt.title("Balanced Performance: Supervised vs. Unsupervised", fontsize=14)
plt.xlabel("Total Training Samples (50/50 Split)", fontsize=12)
plt.ylabel("F1-Score", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('balanced_comparison.png')
plt.show()