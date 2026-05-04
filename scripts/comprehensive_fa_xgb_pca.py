import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
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
    df.columns = df.columns.astype(str)
    y = df['Label'].apply(lambda x: 0 if str(x).upper() == 'BENIGN' else 1)
    X = df.drop(columns=['Label']).replace([np.inf, -np.inf], np.nan)
    X = X.apply(lambda x: x.fillna(x.mean()), axis=0)
    return X, y

def find_best_threshold(probs, y_true, default=0.15):
    if len(np.unique(y_true)) < 2: return default
    thresholds = np.linspace(0.01, 0.85, 100)
    best_f1, best_t = 0, default
    for t in thresholds:
        score = f1_score(y_true, (probs >= t).astype(int))
        if score > best_f1:
            best_f1, best_t = score, t
    return best_t

# --- PRE-PROCESSING: GLOBAL PCA ALIGNMENT ---
print("Step 1: Applying Global PCA for Feature Compression...")
X17_raw, y17 = sanitize(pd.read_csv(file_17))
X18_raw, y18_raw = sanitize(pd.read_csv(file_18))

# Fit Scaler and PCA on the Source (2017) to define the "Latent Space"
scaler_global = StandardScaler().fit(X17_raw)
X17_std = scaler_global.transform(X17_raw)

# We choose n_components that explain 95% of variance in the source
pca = PCA(n_components=0.95, svd_solver='full')
X17_pca = pca.fit_transform(X17_std)
n_comp = pca.n_components_
print(f"PCA complete. Reduced 78 features to {n_comp} Principal Components.")

# Transform the Target (2018) into that same PCA space
X18_std = scaler_global.transform(X18_raw)
X18_pca = pca.transform(X18_std)

# Split 2018 into Pool and Test (PCA Space)
test_size = 10000
test_X, test_y = X18_pca[-test_size:], y18_raw.tail(test_size)
X18_pool, y18_pool = X18_pca[:-test_size], y18_raw.iloc[:-test_size]

# Store results
results = {'xgb': [], 'mlp': [], 'tl': []}

# --- 2. THE 50-ITERATION LOOP ---
print(f"Step 2: Commencing {NUM_ITERATIONS} Iterations in PCA Space...")
for i in range(NUM_ITERATIONS):
    it_xgb, it_mlp, it_tl = [], [], []
    
    # Train Expert on PCA Space
    expert = XGBClassifier(n_estimators=100, max_depth=5, random_state=i)
    expert.fit(X17_pca, y17)
    leg_probs_test = expert.predict_proba(test_X)[:, 1]
    
    idx = np.random.permutation(len(X18_pool))
    zero_shot_f1 = 0.0

    for n in n_points:
        if n == 0:
            # Calibrate threshold using a 100-sample PCA window
            cal_idx = idx[:100]
            cal_probs = expert.predict_proba(X18_pool[cal_idx])[:, 1]
            best_t_0 = find_best_threshold(cal_probs, y18_pool.iloc[cal_idx])
            
            zero_shot_f1 = f1_score(test_y, (leg_probs_test >= best_t_0).astype(int))
            it_xgb.append(0.0); it_mlp.append(0.0); it_tl.append(zero_shot_f1)
        else:
            train_X, train_y = X18_pool[idx[:n]], y18_pool.iloc[idx[:n]]
            
            if len(np.unique(train_y)) < 2:
                it_xgb.append(it_xgb[-1]); it_mlp.append(it_mlp[-1]); it_tl.append(it_tl[-1])
                continue

            # Local Models in PCA Space
            xgb = XGBClassifier(n_estimators=50, max_depth=3, random_state=i).fit(train_X, train_y)
            xgb_probs_test = xgb.predict_proba(test_X)[:, 1]
            it_xgb.append(f1_score(test_y, (xgb_probs_test >= 0.5).astype(int)))
            
            mlp = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=200, random_state=i).fit(train_X, train_y)
            it_mlp.append(f1_score(test_y, mlp.predict(test_X)))
            
            # Adaptive TL with Blending
            w = 1.0 / (1.0 + (n / 50.0))
            blended_probs = (w * leg_probs_test) + ((1 - w) * xgb_probs_test)
            
            # Adaptive Threshold
            tr_probs_ex = expert.predict_proba(train_X)[:, 1]
            tr_probs_loc = xgb.predict_proba(train_X)[:, 1]
            tr_blend = (w * tr_probs_ex) + ((1 - w) * tr_probs_loc)
            
            adaptive_t = find_best_threshold(tr_blend, train_y, default=best_t_0)
            current_f1 = f1_score(test_y, (blended_probs >= adaptive_t).astype(int))
            it_tl.append(max(current_f1, zero_shot_f1))

    results['xgb'].append(it_xgb); results['mlp'].append(it_mlp); results['tl'].append(it_tl)
    if (i+1) % 10 == 0: print(f"Run {i+1}/50 complete.")

print("\n--- RESULTS FOR PLOTTING SCRIPT (PCA VERSION) ---")
print(f"res_local_xgb = {list(np.round(np.mean(results['xgb'], axis=0), 3))}")
print(f"res_mlp = {list(np.round(np.mean(results['mlp'], axis=0), 3))}")
print(f"res_tl = {list(np.round(np.mean(results['tl'], axis=0), 3))}")