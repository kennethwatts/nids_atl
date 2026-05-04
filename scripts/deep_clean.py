import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

def load_and_hard_clean(path):
    print(f"Loading {path}...")
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # List of features that are prone to leakage (IDs, Ports, Timestamps, etc.)
    # We want to remove these to force the model to look at the 'math' of the flow
    drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 
                 'Destination Port', 'Protocol', 'Timestamp', 'External IP']
    
    existing_drops = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drops)
    
    # Ensure Label is cleaned
    df['Label'] = df['Label'].astype(str).str.strip()
    return df

df17 = load_and_hard_clean('cic2017_training_av_lbl_100K_hdr.csv')
df18 = load_and_hard_clean('cic2018_training_av_lbl_100K_hdr.csv')

# Only use common columns found in both files (minus the Label)
features = list(set(df17.columns) & set(df18.columns))
features.remove('Label')

def prep(df):
    y = (df['Label'].str.lower() != 'benign').astype(int).values
    X = df[features].values
    return X, y

# Standardize based on 2017
scaler = StandardScaler().fit(df17[features])
X17, y17 = prep(df17)
X17 = scaler.transform(X17)

# Pre-train Expert
expert_2017 = xgb.XGBClassifier(random_state=42).fit(X17, y17)

# Segregate 2018
all_idx = df18.index.tolist()
np.random.seed(42)
np.random.shuffle(all_idx)
train_pool = all_idx[:5000]
test_pool = all_idx[5000:15000]

df_train_pool = df18.loc[train_pool]
df_test = df18.loc[test_pool]

# Train RF on only 50 samples of 'DDoS' (To ensure it remains a 'specialist')
df_ddos = df_train_pool[(df_train_pool['Label'] == 'Benign') | (df_train_pool['Label'] == 'DDoS')]
df_train = df_ddos.sample(50, random_state=42)

X_train, y_train = prep(df_train)
X_train = scaler.transform(X_train)
X_test, y_test = prep(df_test)
X_test = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)

# EVALUATION
df_test['RF_Hit'] = rf.predict(X_test)
tl_probs = expert_2017.predict_proba(X_test)[:, 1]
df_test['TL_Hit'] = (tl_probs >= np.quantile(tl_probs, 0.90)).astype(int)

print("\n" + "="*50)
print(f"{'Attack Type':<25} | {'RF Detection':<12} | {'TL Detection':<12}")
print("-" * 50)
for label in sorted(df_test['Label'].unique()):
    subset = df_test[df_test['Label'] == label]
    print(f"{label:<25} | {subset['RF_Hit'].mean():>12.1%} | {subset['TL_Hit'].mean():>12.1%}")

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
print("\n--- The 'Cheat' Features ---")
for f in range(5):
    print(f"{f+1}. {features[indices[f]]}: {importances[indices[f]]:.4f}")
    