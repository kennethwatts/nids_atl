import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score
from sklearn.preprocessing import StandardScaler

# 1. LOAD & CLEAN
def load_and_hard_clean(path):
    print(f"Loading {path}...")
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna(0)
    drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 
                 'Destination Port', 'Protocol', 'Timestamp', 'External IP']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df['Label'] = df['Label'].astype(str).str.strip()
    return df

df17 = load_and_hard_clean('cic2017_training_av_lbl_100K_hdr.csv')
df18 = load_and_hard_clean('cic2018_training_av_lbl_100K_hdr.csv')

features = list(set(df17.columns) & set(df18.columns))
features.remove('Label')

# 2. PRE-TRAIN EXPERT (2017)
scaler = StandardScaler().fit(df17[features])
X17 = scaler.transform(df17[features])
y17 = (df17['Label'].str.lower() != 'benign').astype(int).values
expert_2017 = xgb.XGBClassifier(random_state=42).fit(X17, y17)

# 3. BALANCED TRAINING FOR RF (2018)
# We MUST ensure RF sees both Benign and DDoS
benign_18 = df18[df18['Label'] == 'Benign']
attack_18 = df18[df18['Label'] == 'DDoS']

# Sample 50 of each for a fair "Low Data" start
df_train = pd.concat([
    benign_18.sample(min(50, len(benign_18)), random_state=42),
    attack_18.sample(min(50, len(attack_18)), random_state=42)
])

X_train = scaler.transform(df_train[features])
y_train = (df_train['Label'].str.lower() != 'benign').astype(int).values
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

# 4. TEST ON EVERYTHING ELSE
df_test = df18.drop(df_train.index).sample(10000, random_state=42)
X_test = scaler.transform(df_test[features])
y_test = (df_test['Label'].str.lower() != 'benign').astype(int).values

# Predictions
rf_preds = rf.predict(X_test)
tl_probs = expert_2017.predict_proba(X_test)[:, 1]
tl_preds = (tl_probs >= np.quantile(tl_probs, 0.90)).astype(int)

# 5. THE "TRUE" DETECTION REPORT
df_test['RF_Hit'] = rf_preds
df_test['TL_Hit'] = tl_preds

print("\n" + "="*60)
print(f"{'Attack Type':<25} | {'Samples':<8} | {'RF Recall':<10} | {'TL Recall':<10}")
print("-" * 60)
for label in sorted(df_test['Label'].unique()):
    subset = df_test[df_test['Label'] == label]
    # For attacks, we want to know: how many did we catch?
    # For benign, we want to know: how many did we correctly ignore?
    if label == 'Benign':
        rf_rec = (subset['RF_Hit'] == 0).mean() # True Negative Rate
        tl_rec = (subset['TL_Hit'] == 0).mean()
        label_text = "BENIGN (Correct)"
    else:
        rf_rec = (subset['RF_Hit'] == 1).mean() # Recall
        tl_rec = (subset['TL_Hit'] == 1).mean()
        label_text = label

    print(f"{label_text:<25} | {len(subset):<8} | {rf_rec:>9.1%} | {tl_rec:>9.1%}")
print("="*60)