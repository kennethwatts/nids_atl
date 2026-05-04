import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_clean(path):
    print(f"Loading {path}...")
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna(0)
    drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df['Label'] = df['Label'].astype(str).str.lower().str.strip()
    return df

# 1. PRE-TRAIN 2017 EXPERT
df17 = load_clean('cic2017_training_av_lbl_100K_hdr.csv')
features = [f for f in df17.columns if f != 'Label']
scaler = StandardScaler().fit(df17[features])
X17 = scaler.transform(df17[features])
y17 = (df17['Label'] != 'benign').astype(int).values
expert = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42).fit(X17, y17)

# 2. SIMULATE STREAMING DATA (2018)
df18 = load_clean('cic2018_training_av_lbl_100K_hdr.csv')
X18 = scaler.transform(df18[features])
y18 = (df18['Label'] != 'benign').astype(int).values

# 3. CONFIGURATION
window_size = 1000
num_windows = 40 
initial_probs = expert.predict_proba(X17[:2000])[:, 1]
lagged_threshold = np.quantile(initial_probs, 0.90)

stats = []

print(f"\n{'Win':<4} | {'Attacks':<8} | {'F1 (Curr)':<10} | {'F1 (Lag)':<10} | {'Delta':<8}")
print("-" * 50)

for i in range(num_windows):
    start, end = i * window_size, (i + 1) * window_size
    if end > len(X18): break
    
    X_win, y_win = X18[start:end], y18[start:end]
    probs_win = expert.predict_proba(X_win)[:, 1]
    
    # --- CURRENT WINDOW LOGIC ---
    curr_threshold = np.quantile(probs_win, 0.90)
    preds_curr = (probs_win >= curr_threshold).astype(int)
    f1_curr = f1_score(y_win, preds_curr, zero_division=0)
    
    # --- LAGGED WINDOW LOGIC ---
    preds_lag = (probs_win >= lagged_threshold).astype(int)
    f1_lag = f1_score(y_win, preds_lag, zero_division=0)
    
    stats.append({
        'window': i,
        'f1_current': f1_curr,
        'f1_lagged': f1_lag,
        'thresh_diff': abs(curr_threshold - lagged_threshold)
    })
    
    print(f"{i:<4} | {int(np.sum(y_win)):<8} | {f1_curr:<10.3f} | {f1_lag:<10.3f} | {f1_curr - f1_lag:<8.3f}")
    
    # UPDATE lagged threshold for the NEXT iteration
    lagged_threshold = curr_threshold

# 4. EXPORT & PLOT
results_df = pd.DataFrame(stats)
results_df.to_csv("threshold_comparison_results.csv", index=False)

plt.figure(figsize=(12, 6))
plt.plot(results_df['window'], results_df['f1_current'], label='Current-Window Quantile', color='#3498db', linestyle='--')
plt.plot(results_df['window'], results_df['f1_lagged'], label='Lagged-Window Quantile', color='#e67e22', linewidth=2)

plt.title("Comparative Analysis: Current vs. Lagged Adaptive Thresholding", fontsize=14)
plt.xlabel("Sequential Traffic Windows", fontsize=12)
plt.ylabel("F1 Score", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("threshold_strategy_comparison.png")
plt.show()