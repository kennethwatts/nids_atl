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

# 1. SETUP EXPERT (2017)
df17 = load_clean('cic2017_training_av_lbl_100K_hdr.csv')
features = [f for f in df17.columns if f != 'Label']
scaler = StandardScaler().fit(df17[features])
X17 = scaler.transform(df17[features])
y17 = (df17['Label'] != 'benign').astype(int).values
expert = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42).fit(X17, y17)

# 2. SETUP TARGET (2018) - Pick a representative window
df18 = load_clean('cic2018_training_av_lbl_100K_hdr.csv')
window_size = 5000 
X_win = scaler.transform(df18[features][:window_size])
y_win = (df18['Label'][:window_size] != 'benign').astype(int).values
probs_win = expert.predict_proba(X_win)[:, 1]

# 3. CALCULATE F1 ACROSS THRESHOLD RANGE
thresholds = np.linspace(0.01, 0.99, 100)
f1_results = []

# For the "Lagged" comparison, we'll simulate a previous window threshold
# Let's assume the previous window was 'cleaner' or 'noisier' to see the effect
prior_window_probs = expert.predict_proba(X17[:window_size])[:, 1]
lagged_calibrated_thresh = np.quantile(prior_window_probs, 0.90)

for t in thresholds:
    score = f1_score(y_win, (probs_win >= t).astype(int), zero_division=0)
    f1_results.append(score)

# 4. PLOT (Matching your Red/Blue style)
plt.figure(figsize=(10, 6))
plt.plot(thresholds, f1_results, color='#2980b9', linewidth=3, label='F1 Score vs. Threshold (Current)')

# Mark the 'Optimal' point for Current
opt_idx = np.argmax(f1_results)
plt.axvline(thresholds[opt_idx], color='#e74c3c', linestyle='--', alpha=0.7, label=f'Current Optimal ({thresholds[opt_idx]:.2f})')

# Mark the 'Lagged' threshold location
plt.axvline(lagged_calibrated_thresh, color='#f1c40f', linestyle='-', linewidth=2, label=f'Lagged Threshold ({lagged_calibrated_thresh:.2f})')

plt.title("Threshold Sensitivity Analysis: Current vs. Lagged Strategy", fontsize=14, fontweight='bold')
plt.xlabel("Threshold (Quantile / Probability)", fontsize=12)
plt.ylabel("F1 Score", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("threshold_calibration_comparison.png")
plt.show()