import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse

# 1. SETUP & SEEDING (Locking the randomness for thesis consistency)
SEED = 42
np.random.seed(SEED)

parser = argparse.ArgumentParser(description='Micro-Data ML vs TL Comparison')
parser.add_argument('file_2017', help='Path to 2017 dataset')
parser.add_argument('file_2018', help='Path to 2018 dataset')
args = parser.parse_args()

# 2. DATA LOADING
def load_and_binarize(path):
    print(f"Loading {path}...")
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna(0)
    df['Label_Clean'] = df['Label'].astype(str).str.lower().str.strip()
    y = (df['Label_Clean'] != 'benign').astype(int).values
    X = df.drop(columns=['Label', 'Label_Clean']).values
    return X, y

X17, y17 = load_and_binarize(args.file_2017)
X18, y18 = load_and_binarize(args.file_2018)

# TL Scaler: Always looks at the world through 2017 eyes
scaler_tl = StandardScaler().fit(X17)
X17_scaled = scaler_tl.transform(X17)
X18_scaled_tl = scaler_tl.transform(X18)

# Trad Scaler: Learns the 2018 distribution directly (fitted later per sample)

# 3. PRE-TRAIN EXPERT
print("Pre-training Expert on 2017...")
expert_2017 = xgb.XGBClassifier(n_estimators=100, random_state=SEED, eval_metric='logloss')
expert_2017.fit(X17_scaled, y17)

# 4. MICRO-LOOP SETUP (0.01% to 0.10%)
percentages = np.linspace(0.01, 0.1, 10)
f1_rf, f1_xgb, f1_tl = [], [], []

test_size = int(0.2 * len(X18))
indices = np.arange(len(X18))
np.random.shuffle(indices)
test_idx = indices[:test_size]
train_pool_idx = indices[test_size:]

# Consistent test sets
y_test = y18[test_idx]
X_test_tl = X18_scaled_tl[test_idx]

# Expert opinion for TL
expert_test_opinion = expert_2017.predict_proba(X_test_tl)[:, 1].reshape(-1, 1)

print("\nStarting Micro-Data Loop...")
for p in percentages:
    sample_size = max(2, int((p / 100) * len(X18)))
    train_idx = np.random.choice(train_pool_idx, sample_size, replace=False)
    
    X_train_raw = X18[train_idx]
    y_train = y18[train_idx]
    
    # --- Traditional Scaler (Local to 2018) ---
    # We use a fresh scaler for traditional models to avoid 2017 bias
    scaler_trad = StandardScaler().fit(X_train_raw)
    X_train_trad = scaler_trad.transform(X_train_raw)
    X_test_trad = scaler_trad.transform(X18[test_idx])
    
    # --- Traditional RF ---
    rf = RandomForestClassifier(n_estimators=50, random_state=SEED, n_jobs=-1)
    rf.fit(X_train_trad, y_train)
    f1_rf.append(f1_score(y_test, rf.predict(X_test_trad)))
    
    # --- Traditional XGB ---
    trad_xgb = xgb.XGBClassifier(n_estimators=50, random_state=SEED, eval_metric='logloss')
    trad_xgb.fit(X_train_trad, y_train)
    f1_xgb.append(f1_score(y_test, trad_xgb.predict(X_test_trad)))
    
    # --- TL Stacking ---
    X_train_tl = scaler_tl.transform(X_train_raw)
    expert_train_opinion = expert_2017.predict_proba(X_train_tl)[:, 1].reshape(-1, 1)
    
    X_train_aug = np.hstack((X_train_tl, expert_train_opinion))
    X_test_aug = np.hstack((X_test_tl, expert_test_opinion))
    
    tl_stacker = xgb.XGBClassifier(n_estimators=50, random_state=SEED, eval_metric='logloss')
    tl_stacker.fit(X_train_aug, y_train)
    f1_tl.append(f1_score(y_test, tl_stacker.predict(X_test_aug)))
    
    print(f"Data: {p:.3f}% ({sample_size} pts) | RF: {f1_rf[-1]:.3f} | XGB: {f1_xgb[-1]:.3f} | TL: {f1_tl[-1]:.3f}")

# 5. VISUALIZATION
plt.figure(figsize=(10, 6))
plt.plot(percentages, f1_rf, label='RF (Local Scaler)', marker='o', color='#95a5a6')
plt.plot(percentages, f1_xgb, label='XGB (Local Scaler)', marker='s', color='#27ae60')
plt.plot(percentages, f1_tl, label='TL Stacking (2017 Expert)', marker='^', color='#2980b9', linewidth=2.5)

plt.title("Micro-Data Performance Analysis (0.01% - 0.1%)", fontsize=14)
plt.xlabel("Percentage of 2018 Training Data (%)", fontsize=12)
plt.ylabel("F1-Score", fontsize=12)
plt.xticks(percentages)
plt.grid(alpha=0.3, linestyle=':')
plt.legend()
plt.tight_layout()

plt.savefig('micro_data_comparison.png', dpi=300)
print("\nSuccess! Saved as 'micro_data_comparison.png'")
plt.show()