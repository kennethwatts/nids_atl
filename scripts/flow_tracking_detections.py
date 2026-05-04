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

# 1. SETUP EXPERT
df17 = load_clean('cic2017_training_av_lbl_100K_hdr.csv')
features = [f for f in df17.columns if f != 'Label']
scaler = StandardScaler().fit(df17[features])
X17 = scaler.transform(df17[features])
expert = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42).fit(X17, (df17['Label'] != 'benign').astype(int).values)

# 2. DATA STREAM (2018 FLOW SAMPLES)
df18 = load_clean('cic2018_training_av_lbl_100K_hdr.csv')
num_samples = 1500 
X18 = scaler.transform(df18[features][:num_samples]) 
probs_stream = expert.predict_proba(X18)[:, 1]

# 3. CONFIGURATION
window_size = 50
quantile_val = 0.95

thresh_lagged = [np.nan] * window_size 
current_lag_val = np.nan 

for i in range(len(probs_stream)):
    if i >= window_size:
        if i % window_size == 0:
            prior_window = probs_stream[i-window_size : i]
            current_lag_val = np.quantile(prior_window, quantile_val)
        thresh_lagged.append(current_lag_val)

# 4. COLOR CODING THE FLOW SAMPLES
# If lagged thresh is nan (first window), color neutral. Otherwise, color based on detection.
colors = []
for i in range(len(probs_stream)):
    if i < window_size or np.isnan(thresh_lagged[i]):
        colors.append('#3498db') # Default Blue
    elif probs_stream[i] >= thresh_lagged[i]:
        colors.append('#e74c3c') # Detection Red
    else:
        colors.append('#2ecc71') # Filtered Green

# 5. PLOTTING
fig, ax = plt.subplots(figsize=(15, 8))

# Flow Sample Spikes
ax.vlines(range(len(probs_stream)), 0, probs_stream, color=colors, alpha=0.6, label='Flow Sample Scores')

# Threshold Line
line_lag, = ax.plot(range(len(thresh_lagged)), thresh_lagged, color='#2c3e50', linewidth=3, label='Lagged Threshold (Adaptive)')

# LEGENDS
legend1 = ax.legend(handles=[line_lag], loc='upper right', title="Monitoring Logic")
ax.add_artist(legend1)

# Status Legend
det_patch = mlines.Line2D([], [], color='#e74c3c', label='Triggered Alert')
filt_patch = mlines.Line2D([], [], color='#2ecc71', label='Filtered (Below Threshold)')
param_patch = mlines.Line2D([], [], color='none', label=f'Window (W): {window_size} Samples')
quant_patch = mlines.Line2D([], [], color='none', label=f'Quantile (Q): {quantile_val}')
ax.legend(handles=[det_patch, filt_patch, param_patch, quant_patch], loc='upper left', title="System Status & Config")

ax.set_title("Flow-Based Anomaly Tracking: Lagged Adaptive Thresholding", fontsize=14, fontweight='bold')
ax.set_xlabel("Flow Sample Sequence", fontsize=12)
ax.set_ylabel("Anomaly Probability (0.0 - 1.0)", fontsize=12)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("flow_tracking_detections.png", dpi=300)
plt.show()