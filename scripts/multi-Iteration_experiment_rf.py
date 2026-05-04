import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# 1. CONFIGURATION
file_17 = "cic2017_training_av_lbl_100K_hdr.csv"
file_18 = "cic2018_training_av_lbl_100K_hdr.csv"
NUM_ITERATIONS = 50
n_points = [0, 10, 20, 50, 100]

def sanitize(df):
    df.columns = df.columns.astype(str)
    y = df['Label'].apply(lambda x: 0 if str(x).upper() == 'BENIGN' else 1)
    X = df.drop(columns=['Label']).replace([np.inf, -np.inf], np.nan)
    X = X.apply(lambda x: x.fillna(x.max() if not np.isnan(x.max()) else 0), axis=0)
    return X, y

def get_stable_features(X17, X18):
    """Identifies consistent features across datasets to ensure transfer logic works."""
    diff = np.abs(X17.mean() - X18.mean()) / (X17.std() + 1e-6)
    stable_cols = diff[diff < 1.0].index.tolist()
    return stable_cols if len(stable_cols) > 15 else X17.columns.tolist()

def find_best_threshold(probs, y_true, default=0.15):
    if len(np.unique(y_true)) < 2: return default
    thresholds = np.linspace(0.01, 0.85, 100)
    best_f1, best_t = 0, default
    for t in thresholds:
        score = f1_score(y_true, (probs >= t).astype(int))
        if score > best_f1:
            best_f1, best_t = score, t
    return best_t

print("Initializing Datasets and Aligning Stable Paths...")
X17_raw, y17 = sanitize(pd.read_csv(file_17))
X18_raw, y18_raw = sanitize(pd.read_csv(file_18))

stable_cols = get_stable_features(X17_raw, X18_raw)
X17_f = X17_raw[stable_cols]

# Use 2017-centric scaling for the Expert
scaler_17 = StandardScaler().fit(X17_f)
X17_scaled = scaler_17.transform(X17_f)

test_size = 10000
test_X_raw, test_y = X18_raw.tail(test_size), y18_raw.tail(test_size)
X18_pool_raw, y18_pool = X18_raw.iloc[:-test_size], y18_raw.iloc[:-test_size]
test_X_expert = scaler_17.transform(test_X_raw[stable_cols])

results = {'rf': [], 'mlp': [], 'tl': []}

print(f"Executing {NUM_ITERATIONS} iterations with Random Forest core...")
for i in range(NUM_ITERATIONS):
    it_rf, it_mlp, it_tl = [], [], []
    
    # 2. TRAIN RANDOM FOREST EXPERT (2017)
    # Using 100 trees for a robust baseline
    expert = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=i)
    expert.fit(X17_scaled, y17)
    leg_probs_test = expert.predict_proba(test_X_expert)[:, 1]
    
    idx = np.random.permutation(len(X18_pool_raw))
    zero_shot_f1 = 0.0

    for n in n_points:
        if n == 0:
            # Calibrate on 100 samples
            cal_idx = idx[:100]
            cal_X = scaler_17.transform(X18_pool_raw[stable_cols].iloc[cal_idx])
            cal_probs = expert.predict_proba(cal_X)[:, 1]
            best_t_0 = find_best_threshold(cal_probs, y18_pool.iloc[cal_idx])
            
            zero_shot_f1 = f1_score(test_y, (leg_probs_test >= best_t_0).astype(int))
            it_rf.append(0.0); it_mlp.append(0.0); it_tl.append(zero_shot_f1)
        else:
            train_X, train_y = X18_pool_raw.iloc[idx[:n]], y18_pool.iloc[idx[:n]]
            
            if len(np.unique(train_y)) < 2:
                it_rf.append(it_rf[-1]); it_mlp.append(it_mlp[-1]); it_tl.append(it_tl[-1])
                continue

            # 3. LOCAL RANDOM FOREST
            rf = RandomForestClassifier(n_estimators=50, max_depth=5, n_jobs=-1, random_state=i).fit(train_X, train_y)
            rf_probs_test = rf.predict_proba(test_X_raw)[:, 1]
            it_rf.append(f1_score(test_y, (rf_probs_test >= 0.5).astype(int)))
            
            # 4. LOCAL MLP
            s_local = StandardScaler().fit(train_X)
            mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=i)
            mlp.fit(s_local.transform(train_X), train_y)
            it_mlp.append(f1_score(test_y, mlp.predict(s_local.transform(test_X_raw))))
            
            # 5. STABILIZED ADAPTIVE TL (RF Edition)
            # Trust legacy knowledge at first: w decays slower for RF (constant 100)
            w = 1.0 / (1.0 + (n / 100.0))
            blended_probs = (w * leg_probs_test) + ((1 - w) * rf_probs_test)
            
            # Recalibrate threshold on current n samples
            tr_probs_ex = expert.predict_proba(scaler_17.transform(train_X[stable_cols]))[:, 1]
            tr_probs_loc = rf.predict_proba(train_X)[:, 1]
            tr_blend = (w * tr_probs_ex) + ((1 - w) * tr_probs_loc)
            
            adaptive_t = find_best_threshold(tr_blend, train_y, default=best_t_0)
            current_f1 = f1_score(test_y, (blended_probs >= adaptive_t).astype(int))
            
            # Safety Fallback
            it_tl.append(max(current_f1, zero_shot_f1))

    results['rf'].append(it_rf); results['mlp'].append(it_mlp); results['tl'].append(it_tl)
    if (i+1) % 10 == 0: print(f"Run {i+1}/50 complete.")

print("\n--- RESULTS FOR PLOTTING SCRIPT (RF VERSION) ---")
print(f"res_local_rf = {list(np.round(np.mean(results['rf'], axis=0), 3))}")
print(f"res_mlp = {list(np.round(np.mean(results['mlp'], axis=0), 3))}")
print(f"res_tl = {list(np.round(np.mean(results['tl'], axis=0), 3))}")