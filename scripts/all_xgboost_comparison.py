import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. DATA PREP
def load_clean(path):
    print(f"Loading {path}...")
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna(0)
    drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df['Label'] = df['Label'].astype(str).str.lower().str.strip()
    return df

df17 = load_clean('cic2017_training_av_lbl_100K_hdr.csv')
df18 = load_clean('cic2018_training_av_lbl_100K_hdr.csv')
features = [f for f in df17.columns if f != 'Label']

# 2. PRE-TRAINING (2017 EXPERTS)
scaler = StandardScaler().fit(df17[features])
X17 = scaler.transform(df17[features])
y17 = (df17['Label'] != 'benign').astype(int).values

print("Training Legacy XGBoost Model (2017)...")
expert_2017 = xgb.XGBClassifier(n_estimators=100, random_state=42).fit(X17, y17)

# 3. THE EXPERIMENT
labels = [0, 10, 20, 50, 100]
res_local_xgb, res_legacy_adp, res_tl = [], [], []

test_df = df18.sample(20000, random_state=42)
X_test = scaler.transform(test_df[features])
y_test = (test_df['Label'] != 'benign').astype(int).values
probs_17 = expert_2017.predict_proba(X_test)[:, 1]

print("\nRunning All-XGBoost Comparison...")
for n in labels:
    # 1. Legacy Adaptive (2017 Only + Quantile Thresholding)
    thresh_17 = np.quantile(probs_17, 0.9)
    res_legacy_adp.append(f1_score(y_test, (probs_17 >= thresh_17).astype(int)))

    if n == 0:
        res_local_xgb.append(0)
        res_tl.append(res_legacy_adp[-1])
        continue

    # SAFE BALANCED SAMPLING
    pool_18 = df18.drop(test_df.index)
    benign_pool = pool_18[pool_18['Label'] == 'benign']
    attack_pool = pool_18[pool_18['Label'] != 'benign']
    
    n_half = n // 2
    train_df = pd.concat([
        benign_pool.sample(n_half, random_state=42),
        attack_pool.sample(n_half, random_state=42)
    ])
    X_train = scaler.transform(train_df[features])
    y_train = (train_df['Label'] != 'benign').astype(int).values
    
    # 2. Local XGB (2018 Only)
    local_xgb = xgb.XGBClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)
    res_local_xgb.append(f1_score(y_test, local_xgb.predict(X_test)))

    # 3. Adaptive TL (Max-Synergy)
    probs_18 = local_xgb.predict_proba(X_test)[:, 1]
    combined = np.maximum(probs_17, probs_18)
    res_tl.append(f1_score(y_test, (combined >= np.quantile(combined, 0.9)).astype(int)))
    
    print(f"Labels: {n:<3} | Legacy (Adp): {res_legacy_adp[-1]:.3f} | Local XGB: {res_local_xgb[-1]:.3f} | TL (Max): {res_tl[-1]:.3f}")

# 4. PLOT
plt.figure(figsize=(10, 6))
plt.plot(labels, res_legacy_adp, label='Legacy (2017 XGB + Adp)', color='#f39c12', linestyle=':', linewidth=2)
plt.plot(labels, res_local_xgb, label='Local XGB (2018 Only)', color='#e74c3c', linestyle='--')
plt.plot(labels, res_tl, label='Proposed TL (XGB Max-Synergy)', color='#2980b9', linewidth=3)

plt.title("Sample Efficiency: All-XGBoost Platform", fontsize=14)
plt.xlabel("Number of 2018 Samples Added", fontsize=12)
plt.ylabel("F1-Score", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('all_xgboost_comparison.png', dpi=300)
plt.show()