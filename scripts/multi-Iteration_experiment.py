import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings("ignore")

# 1. Configuration
file_17 = "cic2017_training_av_lbl_100K_hdr.csv"
file_18 = "cic2018_training_av_lbl_100K_hdr.csv"
NUM_ITERATIONS = 50
n_points = [0, 10, 20, 50, 100]

def sanitize_data(df):
    df.columns = df.columns.astype(str)
    y = df['Label'].apply(lambda x: 0 if str(x).upper() == 'BENIGN' else 1)
    X = df.drop(columns=['Label']).replace([np.inf, -np.inf], np.nan)
    X = X.apply(lambda x: x.fillna(x.max() if not np.isnan(x.max()) else 0), axis=0)
    return X, y

print("Loading data...")
X17, y17 = sanitize_data(pd.read_csv(file_17))
X18_raw, y18_raw = sanitize_data(pd.read_csv(file_18))

# Use a fixed test set (the final 10% of 2018) for all iterations 
# to ensure the "Security Advantage" is measured against a consistent target.
test_size = 10000
test_X = X18_raw.tail(test_size)
test_y = y18_raw.tail(test_size)
# The "Training Pool" is everything before the test set
X18_pool = X18_raw.iloc[:-test_size]
y18_pool = y18_raw.iloc[:-test_size]

all_xgb, all_mlp, all_tl = [], [], []

print(f"Starting {NUM_ITERATIONS} iterations...")
for i in range(NUM_ITERATIONS):
    # For each iteration, we draw a random subset of 'n' samples from the 2018 pool
    # this simulates different administrators picking different flows to label.
    run_xgb, run_mlp, run_tl = [], [], []
    
    # Train 2017 Expert (Source Knowledge)
    expert = XGBClassifier(n_estimators=100, random_state=i)
    expert.fit(X17, y17)
    # Baseline probabilities from the legacy model
    legacy_probs_test = expert.predict_proba(test_X)[:, 1]

    for n in n_points:
        if n == 0:
            # Zero-Shot Performance
            run_xgb.append(0.0)
            run_mlp.append(0.0)
            run_tl.append(f1_score(test_y, (legacy_probs_test >= 0.08).astype(int)))
        else:
            # Sample n items for training
            idx = np.random.choice(len(X18_pool), n, replace=False)
            train_X, train_y = X18_pool.iloc[idx], y18_pool.iloc[idx]
            
            # --- Local XGB ---
            if len(np.unique(train_y)) < 2:
                run_xgb.append(0.0)
            else:
                xgb = XGBClassifier(n_estimators=50, random_state=i).fit(train_X, train_y)
                run_xgb.append(f1_score(test_y, xgb.predict(test_X)))
            
            # --- Local MLP ---
            if n < 10 or len(np.unique(train_y)) < 2:
                run_mlp.append(0.0)
            else:
                mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=i).fit(train_X, train_y)
                run_mlp.append(f1_score(test_y, mlp.predict(test_X)))
            
            # --- Path Transfer (Adaptive TL) ---
            # We use the legacy logic but 'correct' it using the n samples.
            # This is done by using the legacy probability as a feature for a 2018-specific head.
            legacy_probs_train = expert.predict_proba(train_X)[:, 1].reshape(-1, 1)
            
            if len(np.unique(train_y)) < 2:
                run_tl.append(f1_score(test_y, (legacy_probs_test >= 0.08).astype(int)))
            else:
                # Small 'correction' model to adapt 2017 paths to 2018 realities
                tl_head = XGBClassifier(n_estimators=10, max_depth=2, random_state=i)
                tl_head.fit(legacy_probs_train, train_y)
                refined_preds = tl_head.predict(legacy_probs_test.reshape(-1, 1))
                run_tl.append(f1_score(test_y, refined_preds))

    all_xgb.append(run_xgb)
    all_mlp.append(run_mlp)
    all_tl.append(run_tl)
    if (i+1) % 10 == 0: print(f"Progress: {i+1}/{NUM_ITERATIONS} runs...")

# Averaging (explicitly using numpy arrays to avoid the 'inhomogeneous' error)
final_xgb = np.mean(np.array(all_xgb), axis=0)
final_mlp = np.mean(np.array(all_mlp), axis=0)
final_tl = np.mean(np.array(all_tl), axis=0)

print("\n--- RESULTS FOR PLOTTING SCRIPT ---")
print(f"labels = {n_points}")
print(f"res_local_xgb = {list(np.round(final_xgb, 3))}")
print(f"res_mlp = {list(np.round(final_mlp, 3))}")
print(f"res_tl = {list(np.round(final_tl, 3))}")