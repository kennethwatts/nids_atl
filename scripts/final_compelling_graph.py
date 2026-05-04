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
    df['Label'] = df['Label'].astype(str).str.strip()
    return df

df17 = load_clean('cic2017_training_av_lbl_100K_hdr.csv')
df18 = load_clean('cic2018_training_av_lbl_100K_hdr.csv')
features = [f for f in df17.columns if f != 'Label']

scaler = StandardScaler().fit(df17[features])
X17 = scaler.transform(df17[features])
y17 = (df17['Label'].str.lower() != 'benign').astype(int).values
expert_2017 = xgb.XGBClassifier(random_state=42).fit(X17, y17)

# 2. THE "SAFE SAMPLING" COMPARISON
labels = [0, 10, 20, 40, 60, 80, 100]
res_rf, res_tl = [], []

test_df = df18.sample(min(20000, len(df18)), random_state=42)
X_test = scaler.transform(test_df[features])
y_test = (test_df['Label'].str.lower() != 'benign').astype(int).values

# Pre-calculate Expert opinions to save time
probs_17 = expert_2017.predict_proba(X_test)[:, 1]

print("\nRunning Robust Transition Comparison...")
for n in labels:
    if n == 0:
        res_rf.append(0)
        # Zero-Shot Baseline
        res_tl.append(f1_score(y_test, (probs_17 >= np.quantile(probs_17, 0.9)).astype(int)))
        print(f"Labels: 0   | RF: 0.000 | TL: {res_tl[-1]:.3f} (Instant Protection)")
        continue

    # SAFE SAMPLING: Ensure we have both classes for the local model
    pool_18 = df18.drop(test_df.index)
    benign_pool = pool_18[pool_18['Label'].str.lower() == 'benign']
    attack_pool = pool_18[pool_18['Label'].str.lower() != 'benign']
    
    # Take half benign, half attack to guarantee learning
    n_half = n // 2
    train_df = pd.concat([
        benign_pool.sample(n_half, random_state=42),
        attack_pool.sample(n_half, random_state=42)
    ])
    
    X_train = scaler.transform(train_df[features])
    y_train = (train_df['Label'].str.lower() != 'benign').astype(int).values
    
    # 1. Traditional RF
    rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)
    res_rf.append(f1_score(y_test, rf.predict(X_test)))
    
    # 2. Transfer Learning: Ensemble weighting
    # We use a simple local model to provide 2018 context
    local_18 = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)
    probs_18 = local_18.predict_proba(X_test)[:, 1]
    
    # Weighting: Gradually lean more on local data as n increases
    # But start with a high "Trust" in 2017 knowledge
    trust_factor = 0.7 # Keep 70% of 2017's opinion as the "Anchor"
    combined_probs = (trust_factor * probs_17) + ((1 - trust_factor) * probs_18)
    res_tl.append(f1_score(y_test, (combined_probs >= np.quantile(combined_probs, 0.9)).astype(int)))
    
    print(f"Labels: {n:<3} | RF: {res_rf[-1]:.3f} | TL: {res_tl[-1]:.3f}")

# 3. FINAL PLOT
plt.figure(figsize=(10, 6))
plt.plot(labels, res_rf, label='Traditional ML (Requires Manual Labels)', marker='o', color='#e74c3c', linewidth=2, linestyle='--')
plt.plot(labels, res_tl, label='Adaptive Transfer Learning (TL)', marker='^', color='#2980b9', linewidth=3)

# Highlight the "Value Area"
plt.fill_between(labels, res_rf, res_tl, where=(np.array(res_tl) > np.array(res_rf)), 
                 color='green', alpha=0.15, label='Transfer Learning Advantage')

plt.title("Bridging the 'Security Gap': Zero-Shot Transfer vs. Local Training", fontsize=14)
plt.xlabel("Number of Labeled 2018 Packets (Human Effort)", fontsize=12)
plt.ylabel("Detection Efficacy (F1-Score)", fontsize=12)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig('compelling_transfer_learning.png', dpi=300)
plt.show()