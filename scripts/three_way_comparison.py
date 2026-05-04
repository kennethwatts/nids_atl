import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. DATA PREP
def load_clean(path):
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna(0)
    drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df['Label'] = df['Label'].astype(str).str.strip()
    return df

df17 = load_clean('cic2017_training_av_lbl_100K_hdr.csv')
df18 = load_clean('cic2018_training_av_lbl_100K_hdr.csv')
features = [f for f in df17.columns if f != 'Label']

# 2. PRE-TRAINING (2017 EXPERTS)
scaler = StandardScaler().fit(df17[features])
X17 = scaler.transform(df17[features])
y17 = (df17['Label'].str.lower() != 'benign').astype(int).values

# Legacy RF (Traditional approach: train once, deploy forever)
legacy_rf_2017 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X17, y17)

# Expert XGB for TL
expert_2017 = xgb.XGBClassifier(random_state=42).fit(X17, y17)

# 3. THE EXPERIMENT
labels = [0, 10, 20, 50, 100]
res_local_rf, res_legacy_rf, res_tl = [], [], []

test_df = df18.sample(20000, random_state=42)
X_test = scaler.transform(test_df[features])
y_test = (test_df['Label'].str.lower() != 'benign').astype(int).values

# Pre-calculate Expert opinions
probs_17 = expert_2017.predict_proba(X_test)[:, 1]

print("\nRunning Three-Way Comparison...")
for n in labels:
    # 1. Legacy RF (Always the same - no 2018 data used)
    # Since RF doesn't output probabilities well for thresholding, we use standard predict
    res_legacy_rf.append(f1_score(y_test, legacy_rf_2017.predict(X_test)))

    if n == 0:
        res_local_rf.append(0)
        res_tl.append(f1_score(y_test, (probs_17 >= np.quantile(probs_17, 0.9)).astype(int)))
        continue

    # Sample local 2018 labels
    train_df = df18.drop(test_df.index).sample(n, random_state=42)
    X_train = scaler.transform(train_df[features])
    y_train = (train_df['Label'].str.lower() != 'benign').astype(int).values
    
    # 2. Local RF (2018 Only)
    local_rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)
    res_local_rf.append(f1_score(y_test, local_rf.predict(X_test)))

    # 3. Adaptive TL (Ensemble)
    local_18_xgb = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)
    probs_18 = local_18_xgb.predict_proba(X_test)[:, 1]
    
    # Weighting 2017 knowledge with 2018 local observation
    w = n / 200 # Gradually increase local trust
    combined = ((1-w) * probs_17) + (w * probs_18)
    res_tl.append(f1_score(y_test, (combined >= np.quantile(combined, 0.9)).astype(int)))
    
    print(f"Labels: {n:<3} | Legacy RF: {res_legacy_rf[-1]:.3f} | Local RF: {res_local_rf[-1]:.3f} | TL: {res_tl[-1]:.3f}")

# 4. PLOT
plt.figure(figsize=(10, 6))
plt.plot(labels, res_legacy_rf, label='Legacy RF (Trained on 2017 Only)', color='#f39c12', linestyle=':')
plt.plot(labels, res_local_rf, label='Local RF (Trained on 2018 Only)', color='#e74c3c', linestyle='--')
plt.plot(labels, res_tl, label='Adaptive TL (2017 + 2018)', color='#2980b9', linewidth=3)

plt.title("Legacy Transfer vs. Adaptive Transfer", fontsize=14)
plt.xlabel("Number of 2018 Samples Added", fontsize=12)
plt.ylabel("F1-Score", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('three_way_comparison.png', dpi=300)
plt.show()