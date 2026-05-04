import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

# 1. LOAD & CLEAN
def load_clean(path):
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna(0)
    drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df['Label'] = df['Label'].astype(str).str.lower().str.strip()
    return df

df17 = load_clean('cic2017_training_av_lbl_100K_hdr.csv')
df18 = load_clean('cic2018_training_av_lbl_100K_hdr.csv')
features = [f for f in df17.columns if f != 'Label']

# 2. PRE-TRAIN 2017 (THE ARCHITECT)
scaler = StandardScaler().fit(df17[features])
X17 = scaler.transform(df17[features])
y17 = (df17['Label'] != 'benign').astype(int).values
# Use a shallower tree to capture general behavior rather than specific noise
expert_2017 = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42).fit(X17, y17)

# 3. EXPERIMENT
labels = [0, 10, 20, 50, 100]
res_local, res_legacy, res_tl = [], [], []

test_df = df18.sample(20000, random_state=42)
X_test = scaler.transform(test_df[features])
y_test = (test_df['Label'] != 'benign').astype(int).values

print("\nRunning Behavioral-Synergy Comparison...")
for n in labels:
    # 1. Legacy Baseline (Adaptive)
    p17 = expert_2017.predict_proba(X_test)[:, 1]
    res_legacy.append(f1_score(y_test, (p17 >= np.quantile(p17, 0.9)).astype(int)))

    if n == 0:
        res_local.append(0)
        res_tl.append(res_legacy[-1])
        continue

    # Balanced Sampling
    pool_18 = df18.drop(test_df.index)
    train_df = pd.concat([
        pool_18[pool_18['Label'] == 'benign'].sample(n//2, random_state=42),
        pool_18[pool_18['Label'] != 'benign'].sample(n//2, random_state=42)
    ])
    X_train = scaler.transform(train_df[features])
    y_train = (train_df['Label'] != 'benign').astype(int).values
    
    # 2. Local Only
    local_model = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42).fit(X_train, y_train)
    res_local.append(f1_score(y_test, local_model.predict(X_test)))

    # 3. TRANSFER LEARNING: Decision Path Synergy
    # We use the 2017 model to 'encode' the 2018 data into 2017 logic paths
    # This transforms 70+ features into 50 binary 'Leaf' features
    X_train_leaves = expert_2017.apply(X_train)
    X_test_leaves = expert_2017.apply(X_test)
    
    # Use the 2017 'logic' to train the 2018 'student'
    tl_model = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42).fit(X_train_leaves, y_train)
    res_tl.append(f1_score(y_test, tl_model.predict(X_test_leaves)))
    
    print(f"Labels: {n:<3} | Legacy: {res_legacy[-1]:.3f} | Local: {res_local[-1]:.3f} | TL: {res_tl[-1]:.3f}")

# 4. PLOT
plt.figure(figsize=(10, 6))
plt.plot(labels, res_legacy, label='Legacy (2017)', color='#f39c12', linestyle=':')
plt.plot(labels, res_local, label='Local (2018 Only)', color='#e74c3c', linestyle='--')
plt.plot(labels, res_tl, label='Adaptive TL (Path Transfer)', color='#2980b9', linewidth=3)
plt.title("Sample Efficiency: Decision-Path Transfer Learning", fontsize=14)
plt.xlabel("Number of 2018 Samples", fontsize=12)
plt.ylabel("F1-Score", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('path_transfer_comparison.png')
plt.show()