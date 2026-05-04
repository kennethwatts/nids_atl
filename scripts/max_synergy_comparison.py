import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
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

# 2. PRE-TRAINING
scaler = StandardScaler().fit(df17[features])
X17 = scaler.transform(df17[features])
y17 = (df17['Label'] != 'benign').astype(int).values
expert_2017 = xgb.XGBClassifier(random_state=42).fit(X17, y17)

# 3. EXPERIMENT
labels = [0, 10, 20, 50, 100]
res_local_rf, res_legacy_adaptive, res_tl = [], [], []

test_df = df18.sample(20000, random_state=42)
X_test = scaler.transform(test_df[features])
y_test = (test_df['Label'] != 'benign').astype(int).values
probs_17 = expert_2017.predict_proba(X_test)[:, 1]

print("\nRunning Max-Opinion Comparison...")
for n in labels:
    # 1. Legacy Adaptive (90th percentile)
    thresh_17 = np.quantile(probs_17, 0.9)
    res_legacy_adaptive.append(f1_score(y_test, (probs_17 >= thresh_17).astype(int)))

    if n == 0:
        res_local_rf.append(0)
        res_tl.append(res_legacy_adaptive[-1])
        continue

    # SAFE SAMPLING
    pool_18 = df18.drop(test_df.index)
    benign_pool = pool_18[pool_18['Label'] == 'benign']
    attack_pool = pool_18[pool_18['Label'] != 'benign']
    
    train_df = pd.concat([
        benign_pool.sample(n//2, random_state=42),
        attack_pool.sample(n//2, random_state=42)
    ])
    X_train = scaler.transform(train_df[features])
    y_train = (train_df['Label'] != 'benign').astype(int).values
    
    # 2. Local RF (2018 Only)
    local_rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)
    res_local_rf.append(f1_score(y_test, local_rf.predict(X_test)))

    # 3. TL: Max Opinion
    probs_18 = local_rf.predict_proba(X_test)[:, 1]
    # Synergy: Take the highest risk score from either model
    combined = np.maximum(probs_17, probs_18)
    res_tl.append(f1_score(y_test, (combined >= np.quantile(combined, 0.9)).astype(int)))
    
    print(f"Labels: {n:<3} | Legacy: {res_legacy_adaptive[-1]:.3f} | Local RF: {res_local_rf[-1]:.3f} | TL (Max): {res_tl[-1]:.3f}")

# 4. FINAL PLOT
plt.figure(figsize=(10, 6))
plt.plot(labels, res_legacy_adaptive, label='Legacy (2017 + Adp)', color='#f39c12', linestyle=':', linewidth=2)
plt.plot(labels, res_local_rf, label='Local RF (2018 Only)', color='#e74c3c', linestyle='--')
plt.plot(labels, res_tl, label='Proposed TL (Max-Synergy)', color='#2980b9', linewidth=3)

plt.title("Sample Efficiency: Max-Synergy vs. Traditional Models", fontsize=14)
plt.xlabel("Number of 2018 Samples Added", fontsize=12)
plt.ylabel("F1-Score", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('max_synergy_comparison.png', dpi=300)
plt.show()