import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. LOAD & SANITIZE
def load_and_sanitize(path):
    print(f"Loading {path}...")
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna(0)
    # Remove metadata and potential artifacts
    cheat_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp']
    df = df.drop(columns=[c for c in cheat_cols if c in df.columns])
    df['Label'] = df['Label'].astype(str).str.strip()
    return df

df17 = load_and_sanitize('cic2017_training_av_lbl_100K_hdr.csv')
df18 = load_and_sanitize('cic2018_training_av_lbl_100K_hdr.csv')

features = list(set(df17.columns) & set(df18.columns))
features.remove('Label')

# 2. LOCAL SCALING (The "Alignment" Step)
# This centers 2017 on its own mean and 2018 on its own mean.
scaler17 = StandardScaler().fit(df17[features])
X17_scaled = scaler17.transform(df17[features])
y17 = (df17['Label'].str.lower() != 'benign').astype(int).values

scaler18 = StandardScaler().fit(df18[features]) # Local alignment for 2018

# 3. PRE-TRAIN EXPERT
print("Pre-training Expert on 2017...")
expert_2017 = xgb.XGBClassifier(n_estimators=100, random_state=42).fit(X17_scaled, y17)

# 4. EXPERIMENT LOOP
percentages = [0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
f1_rf, f1_tl = [], []

# Reserve Test Set
test_df = df18.sample(int(0.2 * len(df18)), random_state=42)
X_test = scaler18.transform(test_df[features]) # 2018 testing uses 2018 scaling
y_test = (test_df['Label'].str.lower() != 'benign').astype(int).values

print("\nRunning Aligned Comparison...")

for p in percentages:
    if p == 0:
        f1_rf.append(0)
        # TL Zero-Shot baseline (now using aligned scaling)
        p_tl = expert_2017.predict_proba(X_test)[:, 1]
        f1_tl.append(f1_score(y_test, (p_tl >= np.quantile(p_tl, 0.9)).astype(int)))
        continue

    # Sample training data
    sample_size = max(10, int((p / 100) * len(df18)))
    train_df = df18.drop(test_df.index).sample(sample_size, random_state=42)
    X_train = scaler18.transform(train_df[features]) # 2018 training uses 2018 scaling
    y_train = (train_df['Label'].str.lower() != 'benign').astype(int).values
    
    # --- Traditional RF ---
    rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)
    f1_rf.append(f1_score(y_test, rf.predict(X_test)))
    
    # --- TL Augmented (Stacking) ---
    # The Stacker learns how to map 2017's logic onto 2018's reality
    train_expert_opin = expert_2017.predict_proba(X_train)[:, 1].reshape(-1, 1)
    test_expert_opin = expert_2017.predict_proba(X_test)[:, 1].reshape(-1, 1)
    
    X_train_aug = np.hstack((X_train, train_expert_opin))
    X_test_aug = np.hstack((X_test, test_expert_opin))
    
    tl_stacker = xgb.XGBClassifier(n_estimators=50, random_state=42).fit(X_train_aug, y_train)
    f1_tl.append(f1_score(y_test, tl_stacker.predict(X_test_aug)))
    
    print(f"Data: {p:.3f}% | RF: {f1_rf[-1]:.3f} | TL: {f1_tl[-1]:.3f}")

# 5. FINAL VISUALIZATION
plt.figure(figsize=(10, 6))
plt.plot(percentages, f1_rf, label='Traditional RF (Local 2018)', marker='o', color='#e74c3c', linestyle='--')
plt.plot(percentages, f1_tl, label='TL Stacking (Aligned 2017+2018)', marker='^', color='#2980b9', linewidth=2.5)
plt.title("Sample Efficiency: Aligned Transfer Learning", fontsize=14)
plt.xlabel("Percentage of 2018 Training Data (%)")
plt.ylabel("F1-Score")
plt.grid(alpha=0.3)
plt.legend()
plt.savefig('aligned_rising_curves.png', dpi=300)
plt.show()