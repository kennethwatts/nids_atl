import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_clean(path):
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

# 2. LOAD 2018 DATA
df18 = load_clean('cic2018_training_av_lbl_100K_hdr.csv')
X18 = scaler.transform(df18[features])
y18 = (df18['Label'] != 'benign').astype(int).values

# 3. SETTINGS
window_size = 1000
num_windows = 40
# Initial threshold from 2017 training logic
current_threshold = np.quantile(expert.predict_proba(X17[:2000])[:,1], 0.90)

stats = []

for i in range(num_windows):
    start, end = i * window_size, (i + 1) * window_size
    X_win, y_win = X18[start:end], y18[start:end]
    probs_win = expert.predict_proba(X_win)[:, 1]
    
    # --- CURRENT WINDOW ---
    # We use a more "dynamic" quantile based on the attack density we expect
    curr_threshold = np.quantile(probs_win, 0.85) # Flags top 15%
    f1_curr = f1_score(y_win, (probs_win >= curr_threshold).astype(int), zero_division=0)
    
    # --- LAGGED WINDOW ---
    # At Window 0, we skip the "Lagged" result to omit the 0 point
    f1_lag = None
    if i > 0:
        f1_lag = f1_score(y_win, (probs_win >= lagged_threshold).astype(int), zero_division=0)
    
    stats.append({
        'window': i,
        'f1_current': f1_curr,
        'f1_lagged': f1_lag
    })
    
    lagged_threshold = curr_threshold # Save for next window

# 4. FILTER AND PLOT
results_df = pd.DataFrame(stats)
# Omit the first window where lagged was None
lagged_plot_data = results_df.dropna(subset=['f1_lagged'])

plt.figure(figsize=(12, 6))
plt.plot(results_df['window'], results_df['f1_current'], label='Immediate Quantile (Upper Bound)', color='#3498db', alpha=0.4, linestyle='--')
plt.plot(lagged_plot_data['window'], lagged_plot_data['f1_lagged'], label='Lagged Adaptive Threshold ($t-1$)', color='#e67e22', linewidth=2.5)

plt.title("Stability Analysis: Lagged Threshold Performance", fontsize=14, fontweight='bold')
plt.xlabel("Sequential 1k-Packet Windows", fontsize=12)
plt.ylabel("F1 Score", fontsize=12)
plt.ylim(0, 1)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("refined_lagged_comparison.png")
plt.show()