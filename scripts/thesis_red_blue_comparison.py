import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_clean(path):
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna(0)
    drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df['Label'] = df['Label'].astype(str).str.lower().str.strip()
    return df

# 1. SETUP EXPERT (2017)
df17 = load_clean('cic2017_training_av_lbl_100K_hdr.csv')
features = [f for f in df17.columns if f != 'Label']
scaler = StandardScaler().fit(df17[features])
X17 = scaler.transform(df17[features])
y17 = (df17['Label'] != 'benign').astype(int).values
expert = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42).fit(X17, y17)

# 2. SETUP TARGET (2018)
df18 = load_clean('cic2018_training_av_lbl_100K_hdr.csv')
test_df = df18.sample(20000, random_state=42)
X_test = scaler.transform(test_df[features])
y_test = (test_df['Label'] != 'benign').astype(int).values

# 3. PATH TRANSFER EXPERIMENT
labels_count = [0, 10, 20, 50, 100]
res_local, res_tl = [], []

# Initial Lagged Threshold for TL
initial_probs = expert.predict_proba(X17[:1000])[:, 1]
lagged_thresh = np.quantile(initial_probs, 0.90)

for n in labels_count:
    if n == 0:
        res_local.append(0)
        # TL Zero-Shot use Lagged Threshold
        probs_0 = expert.predict_proba(X_test)[:, 1]
        res_tl.append(f1_score(y_test, (probs_0 >= lagged_thresh).astype(int)))
        continue

    # Training Data for n labels
    pool_18 = df18.drop(test_df.index)
    train_df = pd.concat([
        pool_18[pool_18['Label'] == 'benign'].sample(n//2, random_state=42),
        pool_18[pool_18['Label'] != 'benign'].sample(n//2, random_state=42)
    ])
    X_train = scaler.transform(train_df[features])
    y_train = (train_df['Label'] != 'benign').astype(int).values

    # LOCAL ONLY (Red Line)
    local_model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42).fit(X_train, y_train)
    res_local.append(f1_score(y_test, local_model.predict(X_test)))

    # PATH TRANSFER TL (Blue Line)
    X_train_leaves = expert.apply(X_train)
    X_test_leaves = expert.apply(X_test)
    tl_model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42).fit(X_train_leaves, y_train)
    res_tl.append(f1_score(y_test, tl_model.predict(X_test_leaves)))

# 4. PLOT (Thesis Style)
plt.figure(figsize=(10, 6))
plt.plot(labels_count, res_local, label='Local Training (Traditional)', color='#e74c3c', marker='o', linestyle='--')
plt.plot(labels_count, res_tl, label='Adaptive Path Transfer (Proposed)', color='#2980b9', marker='^', linewidth=3)

# Shade the gain area
plt.fill_between(labels_count, res_local, res_tl, where=(np.array(res_tl) > np.array(res_local)), 
                 color='green', alpha=0.15, label='Reduced Vulnerability Window')

plt.title("Sample Efficiency: Lagged Path Transfer vs. Local Learning", fontsize=14, fontweight='bold')
plt.xlabel("Number of Target Labels ($n$)", fontsize=12)
plt.ylabel("F1 Score", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("thesis_red_blue_comparison.png")
plt.show()