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
num_windows = 50  # Let's run a longer simulation for the CSV
initial_probs = expert.predict_proba(X17[:2000])[:, 1]
current_threshold = np.quantile(initial_probs, 0.90)

# Accumulator for results
window_stats = []

print(f"\n{'Window':<10} | {'Attacks':<10} | {'Threshold':<12} | {'F1 Score':<10}")
print("-" * 55)

for i in range(num_windows):
    start, end = i * window_size, (i + 1) * window_size
    if end > len(X18): break # Safety break
    
    X_win, y_win = X18[start:end], y18[start:end]
    
    # DETECT using prior window's threshold
    probs_win = expert.predict_proba(X_win)[:, 1]
    preds = (probs_win >= current_threshold).astype(int)
    
    # EVALUATE
    score = f1_score(y_win, preds, zero_division=0)
    num_attacks = int(np.sum(y_win))
    
    # STORE metrics for CSV
    window_stats.append({
        'window_index': i,
        'start_packet': start,
        'end_packet': end,
        'attack_count': num_attacks,
        'threshold_used': round(current_threshold, 6),
        'f1_score': round(score, 4)
    })
    
    print(f"{i:<10} | {num_attacks:<10} | {current_threshold:<12.4f} | {score:<10.3f}")
    
    # UPDATE threshold for NEXT window
    current_threshold = np.quantile(probs_win, 0.90)

# 4. SAVE TO LOCAL FILE
results_df = pd.DataFrame(window_stats)
output_filename = "lagged_threshold_results.csv"
results_df.to_csv(output_filename, index=False)
print(f"\n[SUCCESS] Results saved to: {output_filename}")

# 5. QUICK PLOT
plt.figure(figsize=(10, 5))
plt.plot(results_df['window_index'], results_df['f1_score'], color='#16a085', linewidth=2)
plt.fill_between(results_df['window_index'], results_df['f1_score'], alpha=0.2, color='#16a085')
plt.title("Performance Stability: Prior-Window Adaptive Thresholding")
plt.xlabel("Sequential Windows (Time)")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.savefig("stability_plot.png")
plt.show()