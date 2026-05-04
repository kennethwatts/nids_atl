import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings("ignore")

# 1. SETUP
file_17 = "cic2017_training_av_lbl_100K_hdr.csv"
file_18 = "cic2018_training_av_lbl_100K_hdr.csv"
NUM_ITERATIONS = 50
n_points = [0, 10, 20, 50, 100]

def sanitize(df):
    """Cleans data and prepares binary labels."""
    df.columns = df.columns.astype(str)
    y = df['Label'].apply(lambda x: 0 if str(x).upper() == 'BENIGN' else 1)
    X = df.drop(columns=['Label']).replace([np.inf, -np.inf], np.nan)
    X = X.apply(lambda x: x.fillna(x.max() if not np.isnan(x.max()) else 0), axis=0)
    return X, y

def find_best_threshold(probs, y_true):
    """Finds the optimal F1 threshold for the drifted environment."""
    if len(np.unique(y_true)) < 2: return 0.1
    thresholds = np.linspace(0.01, 0.9, 100)
    best_f1, best_t = 0, 0.1
    for t in thresholds:
        score = f1_score(y_true, (probs >= t).astype(int))
        if score > best_f1:
            best_f1, best_t = score, t
    return best_t

print("Loading and Initializing Expert...")
X17_raw, y17 = sanitize(pd.read_csv(file_17))
X18_raw, y18_raw = sanitize(pd.read_csv(file_18))

# Fit Scaler ONLY on 2017 data for the Expert
scaler_17 = StandardScaler().fit(X17_raw)
X17_scaled = scaler_17.transform(X17_raw)

# Prepare 2018 Test/Pool
test_size = 10000
test_X_raw = X18_raw.tail(test_size)
test_y = y18_raw.tail(test_size)
X18_pool_raw = X18_raw.iloc[:-test_size]
y18_pool = y18_raw.iloc[:-test_size]

# Pre-transform 2018 test set using the 2017 scaler for the Expert's view
test_X_expert_view = scaler_17.transform(test_X_raw)

results = {'xgb': [], 'mlp': [], 'tl': []}

print(f"Running {NUM_ITERATIONS} iterations...")
for i in range(NUM_ITERATIONS):
    it_xgb, it_mlp, it_tl = [], [], []
    
    # Train 2017 Expert
    expert = XGBClassifier(n_estimators=100, max_depth=5, random_state=i)
    expert.fit(X17_scaled, y17)
    
    # Get expert probabilities once per iteration
    leg_probs_test = expert.predict_proba(test_X_expert_view)[:, 1]
    
    idx = np.random.permutation(len(X18_pool_raw))
    
    for n in n_points:
        if n == 0:
            # Calibrate threshold on a 500-sample 2018 slice (Expert View)
            cal_idx = idx[:500]
            cal_X = scaler_17.transform(X18_pool_raw.iloc[cal_idx])
            cal_probs = expert.predict_proba(cal_X)[:, 1]
            best_t = find_best_threshold(cal_probs, y18_pool.iloc[cal_idx])
            
            it_xgb.append(0.0)
            it_mlp.append(0.0)
            it_tl.append(f1_score(test_y, (leg_probs_test >= best_t).astype(int)))
        else:
            train_X_raw = X18_pool_raw.iloc[idx[:n]]
            train_y = y18_pool.iloc[idx[:n]]
            
            if len(np.unique(train_y)) < 2:
                it_xgb.append(it_xgb[-1]); it_mlp.append(it_mlp[-1]); it_tl.append(it_tl[-1])
                continue

            # 1. Local XGB (No scaling)
            xgb = XGBClassifier(n_estimators=50, max_depth=3, random_state=i).fit(train_X_raw, train_y)
            local_preds = xgb.predict(test_X_raw)
            it_xgb.append(f1_score(test_y, local_preds))
            
            # 2. Local MLP (Its own local scaler)
            s_local = StandardScaler().fit(train_X_raw)
            mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=i)
            mlp.fit(s_local.transform(train_X_raw), train_y)
            it_mlp.append(f1_score(test_y, mlp.predict(s_local.transform(test_X_raw))))
            
            # 3. Stabilized TL: Weighted Voting Ensemble
            # 'w' starts high (trusting 2017) and decays as n increases
            w = 1.0 / (1.0 + (n / 50.0))
            expert_preds = (leg_probs_test >= best_t).astype(float)
            
            # Combine the two models (Thresholding at 0.5 for binary vote)
            blend = (w * expert_preds) + ((1-w) * local_preds.astype(float))
            it_tl.append(f1_score(test_y, (blend >= 0.5).astype(int)))

    results['xgb'].append(it_xgb)
    results['mlp'].append(it_mlp)
    results['tl'].append(it_tl)
    if (i+1) % 10 == 0: print(f"Progress: {i+1}/50 runs...")

print("\n--- RESULTS FOR PLOTTING SCRIPT ---")
print(f"res_local_xgb = {list(np.round(np.mean(results['xgb'], axis=0), 3))}")
print(f"res_mlp = {list(np.round(np.mean(results['mlp'], axis=0), 3))}")
print(f"res_tl = {list(np.round(np.mean(results['tl'], axis=0), 3))}")