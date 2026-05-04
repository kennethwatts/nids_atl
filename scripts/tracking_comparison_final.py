import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def load_clean(path):
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna(0)
    drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df['Label'] = df['Label'].astype(str).str.lower().str.strip()
    return df

# 1. SETUP EXPERT (2017)
df17 = load_clean('cic2017_training_av_lbl_100K_hdr.csv')
features = [f for f in df17.columns if f != 'Label']
scaler = StandardScaler().fit(df17[features])
X17 = scaler.transform(df17[features])
expert = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42).fit(X17, (df17['Label'] != 'benign').astype(int).values)

# 2. STREAMING 2018 DATA
df18 = load_clean('cic2018_training_av_lbl_100K_hdr.csv')
num_samples = 2000 
X18 = scaler.transform(df18[features][:num_samples]) 
probs_stream = expert.predict_proba(X18)[:, 1]

# 3. TRACKING CONFIGURATION
window_size = 50
quantile_val = 0.95

thresh_lagged = [np.nan] * window_size # Omit first window data
thresh_current = []
current_lag_val = np.nan 

for i in range(len(probs_stream)):
    window_start = (i // window_size) * window_size
    window_end = window_start + window_size
    
    # CURRENT WINDOW THRESHOLD
    win_data_curr = probs_stream[window_start:window_end]
    thresh_current.append(np.quantile(win_data_curr, quantile_val))
    
    # LAGGED WINDOW THRESHOLD (Starts at index 'window_size')
    if i >= window_size:
        if i % window_size == 0:
            prior_window = probs_stream[i-window_size : i]
            current_lag_val = np.quantile(prior_window, quantile_val)
        thresh_lagged.append(current_lag_val)

# 4. PLOTTING
fig, ax = plt.subplots(figsize=(15, 8))

# Probabilities
p_dots = ax.vlines(range(len(probs_stream)), 0, probs_stream, color='#3498db', alpha=0.3, label='Packet Probabilities')

# Current Window Threshold (The "Oracle")
line_curr, = ax.plot(range(len(probs_stream)), thresh_current, color='#9b59b6', linestyle='--', linewidth=1.5, label='Current Window Strategy')

# Lagged Window Threshold (The Practical)
line_lag, = ax.plot(range(len(thresh_lagged)), thresh_lagged, color='#e74c3c', linewidth=2.5, label='Lagged Window Strategy')

# CREATE TWO LEGENDS
# Legend 1: Methods
legend1 = ax.legend(handles=[line_curr, line_lag], loc='upper right', title="Tracking Methodology", frameon=True, shadow=True)
ax.add_artist(legend1)

# Legend 2: Configuration Parameters (Proxy artists)
param_patch = mlines.Line2D([], [], color='none', label=f'Window Size (W): {window_size}')
quantile_patch = mlines.Line2D([], [], color='none', label=f'Quantile (Q): {quantile_val}')
ax.legend(handles=[param_patch, quantile_patch], loc='upper left', title="System Configuration", frameon=True)

# Formatting
ax.set_title("Dynamic Threshold Tracking: Real-Time vs. Lagged Quantile Adaptation", fontsize=14, fontweight='bold')
ax.set_xlabel("Packet Sequence", fontsize=12)
ax.set_ylabel("Detection Threshold / Probability", fontsize=12)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("tracking_comparison_final.png", dpi=300)
plt.show()