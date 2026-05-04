import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
y17 = (df17['Label'] != 'benign').astype(int).values
expert = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42).fit(X17, y17)

# 2. STREAMING 2018 DATA
df18 = load_clean('cic2018_training_av_lbl_100K_hdr.csv')
X18 = scaler.transform(df18[features][:5000]) # Look at first 5k packets
probs_stream = expert.predict_proba(X18)[:, 1]

# 3. TRACKING LOGIC
window_size = 500
threshold_history = []
# Initialize with a reasonable starting point
current_thresh = 0.2 

for i in range(len(probs_stream)):
    # Every 'window_size' packets, update the threshold using the PRIOR window
    if i > 0 and i % window_size == 0:
        prior_window = probs_stream[i-window_size : i]
        current_thresh = np.quantile(prior_window, 0.90) # Top 10% logic
    
    threshold_history.append(current_thresh)

# 4. PLOT (Matching your description)
plt.figure(figsize=(15, 6))

# Blue vertical lines for model probabilities
plt.vlines(range(len(probs_stream)), 0, probs_stream, color='#3498db', alpha=0.4, label='Model Probability (Packet Score)')

# Red horizontal-ish line for Adaptive Threshold
plt.plot(range(len(probs_stream)), threshold_history, color='#e74c3c', linewidth=2, label='Lagged Adaptive Threshold')

plt.title("Dynamic Threshold Tracking: Temporal Adaptation (Lagged Quantile)", fontsize=14, fontweight='bold')
plt.xlabel("Packet Sequence", fontsize=12)
plt.ylabel("Probability / Detection Sensitivity", fontsize=12)
plt.legend(loc='upper right')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("dynamic_threshold_tracking.png", dpi=300)
plt.show()