import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse

# 1. SETUP & SEEDING (Seed 42 for consistency)
SEED = 42
np.random.seed(SEED)

parser = argparse.ArgumentParser(description='High-Res ML vs TL Comparison')
parser.add_argument('file_2017', help='Path to 2017 dataset')
parser.add_argument('file_2018', help='Path to 2018 dataset')
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

scaler = StandardScaler().fit(X17)
X17_scaled = scaler.transform(X17)
X18_scaled = scaler.transform(X18)

# 3. PRE-TRAINING THE EXPERT (2017)
print("Pre-training Expert model on 2017 data...")
expert_2017 = xgb.XGBClassifier(n_estimators=100, random_state=SEED, eval_metric='logloss')
expert_2017.fit(X17_scaled, y17)

# 4. EXPERIMENT SETUP (0.1% to 1.0%)
# We iterate in increments of 0.1%
percentages = np.linspace(0.1, 1.0, 10)
f1_rf, f1_xgb, f1_tl = [], [], []

test_size = int(0.2 * len(X18))
indices = np.arange(len(X18))
np.random.shuffle(indices)
test_idx = indices[:test_size]
train_pool_idx = indices[test_size:]

X_test, y_test = X18_scaled[test_idx], y18[test_idx]
expert_test_opinion = expert_2017.predict_proba(X_test)[:, 1].reshape(-1, 1)

print("\nStarting High-Resolution Loop...")

for p in percentages:
    # Calculate very small sample sizes
    sample_size = int((p / 100) * len(X18))
    train_idx = np.random.choice(train_pool_idx, sample_size, replace=False)
    
    X_train, y_train = X18_scaled[train_idx], y18[train_idx]
    
    # --- Traditional RF ---
    rf = RandomForestClassifier(n_estimators=50, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    f1_rf.append(f1_score(y_test, rf.predict(X_test)))
    
    # --- Traditional XGB ---
    trad_xgb = xgb.XGBClassifier(n_estimators=50, random_state=SEED, eval_metric='logloss')
    trad_xgb.fit(X_train, y_train)
    f1_xgb.append(f1_score(y_test, trad_xgb.predict(X_test)))
    
    # --- TL Stacking ---
    expert_train_opinion = expert_2017.predict_proba(X_train)[:, 1].reshape(-1, 1)
    X_train_aug = np.hstack((X_train, expert_train_opinion))
    X_test_aug = np.hstack((X_test, expert_test_opinion))
    
    tl_stacker = xgb.XGBClassifier(n_estimators=50, random_state=SEED, eval_metric='logloss')
    tl_stacker.fit(X_train_aug, y_train)
    f1_tl.append(f1_score(y_test, tl_stacker.predict(X_test_aug)))
    
    print(f"Data: {p:.1f}% | RF: {f1_rf[-1]:.3f} | XGB: {f1_xgb[-1]:.3f} | TL: {f1_tl[-1]:.3f}")

# 5. VISUALIZATION
plt.figure(figsize=(10, 6))
plt.plot(percentages, f1_rf, label='Random Forest (Scratch)', marker='o', color='#95a5a6')
plt.plot(percentages, f1_xgb, label='XGBoost (Scratch)', marker='s', color='#27ae60')
plt.plot(percentages, f1_tl, label='TL Stacking (2017 Expert)', marker='^', color='#2980b9', linewidth=2.5)

plt.title("TL Advantage at Extreme Low Data (0.1% - 1.0%)", fontsize=14)
plt.xlabel("Percentage of 2018 Training Data (%)", fontsize=12)
plt.ylabel("F1-Score", fontsize=12)
plt.xticks(percentages)
plt.grid(alpha=0.3, linestyle=':')
plt.legend()
plt.tight_layout()

plt.savefig('tl_high_res_advantage.png', dpi=300)
print("\nSuccess! Saved as 'tl_high_res_advantage.png'")
plt.show()