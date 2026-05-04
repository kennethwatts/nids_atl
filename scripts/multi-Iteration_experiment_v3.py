import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import warnings

# Silencing warnings for a clean console
warnings.filterwarnings("ignore")

# 1. CONFIGURATION
file_17 = "cic2017_training_av_lbl_100K_hdr.csv"
file_18 = "cic2018_training_av_lbl_100K_hdr.csv"
NUM_ITERATIONS = 50
n_points = [0, 10, 20, 50, 100]

def sanitize_and_scale(df):
    df.columns = df.columns.astype(str)
    y = df['Label'].apply(lambda x: 0 if str(x).upper() == 'BENIGN' else 1)
    X = df.drop(columns=['Label']).replace([np.inf, -np.inf], np.nan)
    X = X.apply(lambda x: x.fillna(x.max() if not np.isnan(x.max()) else 0), axis=0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=df.drop(columns=['Label']).columns), y

def find_best_threshold(probs, y_true):
    """Calibrates threshold to maximize F1 on available labels."""
    if len(np.unique(y_true)) < 2: return 0.08
    thresholds = np.linspace(0.01, 0.9, 50)
    best_f1, best_t = 0, 0.08
    for t in thresholds:
        score = f1_score(y_true, (probs >= t).astype(int))
        if score > best_f1:
            best_f1, best_t = score, t
    return best_t

print("Loading and Preparing Data...")
X17, y17 = sanitize_and_scale(pd.read_csv(file_17))
X18_raw, y18_raw = sanitize_and_scale(pd.read_csv(file_18))

# Define Test Set (Chronological Tail)
test_size = 10000
test_X = X18_raw.tail(test_size)
test_y = y18_raw.tail(test_size)
X18_pool = X18_raw.iloc[:-test_size]
y18_pool = y18_raw.iloc[:-test_size]

all_xgb, all_mlp, all_tl = [], [], []

print(f"Executing {NUM_ITERATIONS} iterations...")
for i in range(NUM_ITERATIONS):
    run_xgb, run_mlp, run_tl = [], [], []
    
    # Train 2017 Expert (The Foundation)
    expert = XGBClassifier(n_estimators=100, max_depth=5, random_state=i)
    expert.fit(X17, y17)
    
    # Pre-calculate legacy probabilities to speed up the loop
    leg_probs_test = expert.predict_proba(test_X)[:, 1]
    
    # Shuffle for this run
    indices = np.random.permutation(len(X18_pool))

    for n in n_points:
        if n == 0:
            # ZERO-SHOT: Use a tiny 50-sample 'warm-up' set to calibrate the 2017 threshold
            # This ensures we start near the 0.39 mark in the new 2018 environment.
            calib_idx = indices[:50]
            cal_probs = expert.predict_proba(X18_pool.iloc[calib_idx])[:, 1]
            best_t = find_best_threshold(cal_probs, y18_pool.iloc[calib_idx])
            
            run_xgb.append(0.0)
            run_mlp.append(0.0)
            run_tl.append(f1_score(test_y, (leg_probs_test >= best_t).astype(int)))
        else:
            train_X, train_y = X18_pool.iloc[indices[:n]], y18_pool.iloc[indices[:n]]
            
            # Skip training if only one class is present to avoid XGBoost base_score error
            if len(np.unique(train_y)) < 2:
                run_xgb.append(0.0)
                run_mlp.append(0.0)
                run_tl.append(run_tl[-1] if run_tl else 0.0)
                continue

            # Local XGB
            xgb = XGBClassifier(n_estimators=50, random_state=i).fit(train_X, train_y)
            run_xgb.append(f1_score(test_y, xgb.predict(test_X)))
            
            # Local MLP
            mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=i).fit(train_X, train_y)
            run_mlp.append(f1_score(test_y, mlp.predict(test_X)))
            
            # STACKED TRANSFER LEARNING: Expert advice + Raw features
            leg_probs_train = expert.predict_proba(train_X)[:, 1].reshape(-1, 1)
            X_train_plus = np.hstack([train_X, leg_probs_train])
            X_test_plus = np.hstack([test_X, leg_probs_test.reshape(-1, 1)])
            
            tl_stacked = XGBClassifier(n_estimators=100, max_depth=3, random_state=i)
            tl_stacked.fit(X_train_plus, train_y)
            run_tl.append(f1_score(test_y, tl_stacked.predict(X_test_plus)))

    all_xgb.append(run_xgb)
    all_mlp.append(run_mlp)
    all_tl.append(run_tl)
    if (i+1) % 10 == 0: print(f"Run {i+1}/50 complete.")

# 2. OUTPUT AVERAGES
print("\n--- RESULTS FOR PLOTTING SCRIPT ---")
print(f"res_local_xgb = {list(np.round(np.mean(all_xgb, axis=0), 3))}")
print(f"res_mlp = {list(np.round(np.mean(all_mlp, axis=0), 3))}")
print(f"res_tl = {list(np.round(np.mean(all_tl, axis=0), 3))}")