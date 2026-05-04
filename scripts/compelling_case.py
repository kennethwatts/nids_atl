import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. SETUP
SEED = 42
def load_clean(path):
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna(0)
    # Remove metadata to avoid cheating
    drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df['Label'] = df['Label'].astype(str).str.strip()
    return df

df17 = load_clean('cic2017_training_av_lbl_100K_hdr.csv')
df18 = load_clean('cic2018_training_av_lbl_100K_hdr.csv')
features = [f for f in df17.columns if f != 'Label']

# 2. PRE-TRAIN (2017)
scaler = StandardScaler().fit(df17[features])
X17 = scaler.transform(df17[features])
y17 = (df17['Label'].str.lower() != 'benign').astype(int).values
expert_2017 = xgb.XGBClassifier(random_state=SEED).fit(X17, y17)

# 3. THE COST-BENEFIT LOOP
label_counts = [0, 10, 20, 40, 60, 80, 100]
res_rf, res_tl = [], []

test_df = df18.sample(20000, random_state=SEED)
y_test = (test_df['Label'].str.lower() != 'benign').astype(int).values
X_test = scaler.transform(test_df[features])

print("\nRunning Label-Cost Comparison...")
for n in label_counts:
    if n == 0:
        # Traditional ML is literally impossible at 0 labels
        res_rf.append(0)
        # TL Expert uses 2017 knowledge + Adaptive Quantile (0 labels needed)
        probs = expert_2017.predict_proba(X_test)[:, 1]
        # We assume 10% of traffic is malicious for the threshold
        res_tl.append(f1_score(y_test, (probs >= np.quantile(probs, 0.90)).astype(int)))
        print(f"Labels: {n:<3} | RF: 0.000 | TL: {res_tl[-1]:.3f} (Zero-Shot Protection)")
        continue

    # Sample n labels from 2018
    train_df = df18.drop(test_df.index).sample(n)
    X_train = scaler.transform(train_df[features])
    y_train = (train_df['Label'].str.lower() != 'benign').astype(int).values
    
    # RF (Supervised)
    rf = RandomForestClassifier(n_estimators=50).fit(X_train, y_train)
    res_rf.append(f1_score(y_test, rf.predict(X_test)))
    
    # TL (Supervised Fine-tuning)
    # Using the stacking method from before
    train_opin = expert_2017.predict_proba(X_train)[:, 1].reshape(-1, 1)
    test_opin = expert_2017.predict_proba(X_test)[:, 1].reshape(-1, 1)
    tl_model = xgb.XGBClassifier(n_estimators=50).fit(np.hstack((X_train, train_opin)), y_train)
    res_tl.append(f1_score(y_test, tl_model.predict(np.hstack((X_test, test_opin)))))
    
    print(f"Labels: {n:<3} | RF: {res_rf[-1]:.3f} | TL: {res_tl[-1]:.3f}")

# 4. PLOT
plt.figure(figsize=(10, 6))
plt.plot(label_counts, res_rf, label='Traditional ML (Requires Manual Labels)', marker='o', color='#e74c3c', linewidth=2)
plt.plot(label_counts, res_tl, label='Transfer Learning (Pre-trained Protection)', marker='^', color='#2980b9', linewidth=3)
plt.axhline(y=res_tl[0], color='#2980b9', linestyle='--', alpha=0.5, label='TL Zero-Label Baseline')

plt.title("The Cost of Security: TL vs. Traditional ML", fontsize=14)
plt.xlabel("Number of Manual Labels Required from 2018 Data", fontsize=12)
plt.ylabel("Detection Performance (F1-Score)", fontsize=12)
plt.grid(alpha=0.3)
plt.legend()
plt.savefig('compelling_case.png')
plt.show()