import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse

# 1. SETUP & SEEDING
SEED = 42
np.random.seed(SEED)

parser = argparse.ArgumentParser(description='Unsupervised TL vs Supervised Traditional ML')
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

# Fit scaler based on the 2017 pre-training environment
scaler_2017 = StandardScaler().fit(X17)
X17_scaled = scaler_2017.transform(X17)
X18_scaled = scaler_2017.transform(X18)

# 3. PRE-TRAIN THE 2017 EXPERT
print("Pre-training 2017 Expert...")
expert_2017 = xgb.XGBClassifier(n_estimators=100, random_state=SEED, eval_metric='logloss')
expert_2017.fit(X17_scaled, y17)

# 4. EXPERIMENT SETUP
# We compare labels available (10 to 100)
percentages = np.linspace(0.01, 0.1, 10)
f1_supervised_rf = []
f1_unsupervised_tl = []

test_size = int(0.2 * len(X18))
indices = np.arange(len(X18))
np.random.shuffle(indices)
test_idx = indices[:test_size]
train_pool_idx = indices[test_size:]

X_test = X18_scaled[test_idx]
y_test = y18[test_idx]

print("\nRunning Comparison...")
for p in percentages:
    sample_size = max(2, int((p / 100) * len(X18)))
    train_idx = np.random.choice(train_pool_idx, sample_size, replace=False)
    
    # --- SUPERVISED TRADITIONAL ML (Needs Labels) ---
    X_train_trad = X18_scaled[train_idx]
    y_train_trad = y18[train_idx]
    
    rf = RandomForestClassifier(n_estimators=50, random_state=SEED)
    rf.fit(X_train_trad, y_train_trad)
    f1_supervised_rf.append(f1_score(y_test, rf.predict(X_test)))

    # --- UNSUPERVISED TL (Does NOT need 2018 labels) ---
    # It uses the 2017 Expert + Adaptive Quantile Thresholding
    # Note: It only uses the 2018 data to calculate the threshold, not for training!
    expert_probs_test = expert_2017.predict_proba(X_test)[:, 1]
    
    # We simulate "Unsupervised" by taking the 95th percentile of the current 2018 window
    # No y_train labels are used here at all.
    current_2018_window_probs = expert_2017.predict_proba(X_test[:sample_size])[:, 1]
    thresh = np.quantile(current_2018_window_probs, 0.95)
    
    tl_preds = (expert_probs_test >= thresh).astype(int)
    f1_unsupervised_tl.append(f1_score(y_test, tl_preds))
    
    print(f"Sample Size: {sample_size:<4} | Supervised RF: {f1_supervised_rf[-1]:.3f} | Unsupervised TL: {f1_unsupervised_tl[-1]:.3f}")

# 5. VISUALIZATION
plt.figure(figsize=(10, 6))
plt.plot(percentages, f1_supervised_rf, label='Supervised Random Forest (Requires 2018 Labels)', marker='o', color='#e74c3c')
plt.plot(percentages, f1_unsupervised_tl, label='Unsupervised TL Expert (No 2018 Labels Used)', marker='^', color='#2980b9', linewidth=2.5)

plt.title("Label-Dependency Comparison: Supervised vs. Unsupervised Transfer", fontsize=14)
plt.xlabel("Percentage of 2018 Data Available (%)", fontsize=12)
plt.ylabel("F1-Score", fontsize=12)
plt.grid(alpha=0.3, linestyle=':')
plt.legend()
plt.tight_layout()

plt.savefig('unsupervised_vs_supervised.png', dpi=300)
print("\nGraph saved as 'unsupervised_vs_supervised.png'")
plt.show()