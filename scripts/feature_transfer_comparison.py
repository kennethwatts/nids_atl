import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. LOAD DATA
def load_clean(path):
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna(0)
    drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df['Label'] = df['Label'].astype(str).str.lower().str.strip()
    return df

df17 = load_clean('cic2017_training_av_lbl_100K_hdr.csv')
df18 = load_clean('cic2018_training_av_lbl_100K_hdr.csv')
features = [f for f in df17.columns if f != 'Label']

# 2. PRE-TRAIN 2017
scaler = StandardScaler().fit(df17[features])
X17 = scaler.transform(df17[features])
y17 = (df17['Label'] != 'benign').astype(int).values
expert_2017 = xgb.XGBClassifier(n_estimators=100, random_state=42).fit(X17, y17)

# 3. EXPERIMENT
labels = [0, 10, 20, 50, 100]
res_local, res_legacy, res_tl = [], [], []

test_df = df18.sample(20000, random_state=42)
X_test = scaler.transform(test_df[features])
y_test = (test_df['Label'] != 'benign').astype(int).values
test_expert_opin = expert_2017.predict_proba(X_test)[:, 1].reshape(-1, 1)

print("\nRunning Feature-Transfer Comparison...")
for n in labels:
    # Legacy Baseline
    p17 = expert_2017.predict_proba(X_test)[:, 1]
    res_legacy.append(f1_score(y_test, (p17 >= np.quantile(p17, 0.9)).astype(int)))

    if n == 0:
        res_local.append(0)
        res_tl.append(res_legacy[-1])
        continue

    # Sample 2018 (Balanced)
    pool_18 = df18.drop(test_df.index)
    train_df = pd.concat([
        pool_18[pool_18['Label'] == 'benign'].sample(n//2, random_state=42),
        pool_18[pool_18['Label'] != 'benign'].sample(n//2, random_state=42)
    ])
    X_train = scaler.transform(train_df[features])
    y_train = (train_df['Label'] != 'benign').astype(int).values
    
    # 1. Local Only
    local_model = xgb.XGBClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)
    res_local.append(f1_score(y_test, local_model.predict(X_test)))

    # 2. Transfer Learning (Expert Opinion as a Feature)
    train_expert_opin = expert_2017.predict_proba(X_train)[:, 1].reshape(-1, 1)
    
    # Stack the original features WITH the 2017 Expert's prediction
    X_train_tl = np.hstack((X_train, train_expert_opin))
    X_test_tl = np.hstack((X_test, test_expert_opin))
    
    tl_model = xgb.XGBClassifier(n_estimators=50, random_state=42).fit(X_train_tl, y_train)
    res_tl.append(f1_score(y_test, tl_model.predict(X_test_tl)))
    
    print(f"Labels: {n:<3} | Legacy: {res_legacy[-1]:.3f} | Local: {res_local[-1]:.3f} | TL: {res_tl[-1]:.3f}")

# 4. PLOT
plt.figure(figsize=(10, 6))
plt.plot(labels, res_legacy, label='Legacy (2017)', color='#f39c12', linestyle=':')
plt.plot(labels, res_local, label='Local (2018 Only)', color='#e74c3c', linestyle='--')
plt.plot(labels, res_tl, label='Adaptive TL (Feature Transfer)', color='#2980b9', linewidth=3)
plt.title("Sample Efficiency: Feature-Augmented Transfer Learning", fontsize=14)
plt.xlabel("Number of 2018 Samples", fontsize=12)
plt.ylabel("F1-Score", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('feature_transfer_comparison.png', dpi=300)
plt.show()