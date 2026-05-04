import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# 1. CONFIGURATION
file_17 = "cic2017_training_av_lbl_100K_hdr.csv"
file_18 = "cic2018_training_av_lbl_100K_hdr.csv"
NUM_ITERATIONS = 50
n_points = [0, 10, 20, 50, 100]

def sanitize_data(df):
    """Cleans INF/NaN and converts labels to binary."""
    df.columns = df.columns.astype(str)
    # Binary Mapping: Benign vs All Attacks
    y = df['Label'].apply(lambda x: 0 if str(x).upper() == 'BENIGN' else 1)
    X = df.drop(columns=['Label']).replace([np.inf, -np.inf], np.nan)
    # Fill NaN with column max to preserve intensity information
    X = X.apply(lambda x: x.fillna(x.max() if not np.isnan(x.max()) else 0), axis=0)
    return X, y

def find_best_threshold(probs, y_true):
    """
    Stabilizes Path Transfer by finding the F1-optimal threshold 
    on the small 'n' sample set.
    """
    if len(np.unique(y_true)) < 2: 
        return 0.08 # Fallback to 2017 baseline if sample is non-diverse
    
    thresholds = np.linspace(0.01, 0.90, 30)
    best_f1, best_t = 0, 0.08
    for t in thresholds:
        score = f1_score(y_true, (probs >= t).astype(int))
        if score > best_f1:
            best_f1 = score
            best_t = t
    return best_t

print("Loading and Sanitizing Data...")
X17, y17 = sanitize_data(pd.read_csv(file_17))
X18_raw, y18_raw = sanitize_data(pd.read_csv(file_18))

# Define fixed test set (chronological tail)
test_size = 10000
test_X = X18_raw.tail(test_size)
test_y = y18_raw.tail(test_size)
X18_pool = X18_raw.iloc[:-test_size]
y18_pool = y18_raw.iloc[:-test_size]

all_xgb, all_mlp, all_tl = [], [], []

print(f"Starting {NUM_ITERATIONS} iterations...")
for i in range(NUM_ITERATIONS):
    run_xgb, run_mlp, run_tl = [], [], []
    
    # Train 2017 Expert for this run
    expert = XGBClassifier(n_estimators=100, random_state=i)
    expert.fit(X17, y17)
    legacy_probs_test = expert.predict_proba(test_X)[:, 1]

    # Generate random indices for the 'n' sample draw in this iteration
    # This ensures a unique 'cold start' experience for every run
    indices = np.random.permutation(len(X18_pool))

    for n in n_points:
        if n == 0:
            run_xgb.append(0.0)
            run_mlp.append(0.0)
            run_tl.append(f1_score(test_y, (legacy_probs_test >= 0.08).astype(int)))
        else:
            # Draw n samples from the pool
            train_idx = indices[:n]
            train_X, train_y = X18_pool.iloc[train_idx], y18_pool.iloc[train_idx]
            
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
            
            # --- Adaptive TL (Threshold Optimization) ---
            # Get probabilities on the small training set to find the best threshold
            probs_on_n = expert.predict_proba(train_X)[:, 1]
            adaptive_t = find_best_threshold(probs_on_n, train_y)
            
            # Apply adapted threshold to test set
            preds = (legacy_probs_test >= adaptive_t).astype(int)
            run_tl.append(f1_score(test_y, preds))

    all_xgb.append(run_xgb)
    all_mlp.append(run_mlp)
    all_tl.append(run_tl)
    if (i+1) % 10 == 0:
        print(f"Completed iteration {i+1}/{NUM_ITERATIONS}")

# 2. CALCULATE FINAL MEANS
final_xgb = np.mean(np.array(all_xgb), axis=0)
final_mlp = np.mean(np.array(all_mlp), axis=0)
final_tl = np.mean(np.array(all_tl), axis=0)

print("\n--- RESULTS FOR PLOTTING SCRIPT ---")
print(f"labels = {n_points}")
print(f"res_local_xgb = {list(np.round(final_xgb, 3))}")
print(f"res_mlp = {list(np.round(final_mlp, 3))}")
print(f"res_tl = {list(np.round(final_tl, 3))}")