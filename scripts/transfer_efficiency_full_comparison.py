import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from numpy import trapz 
import warnings

warnings.filterwarnings("ignore")

# 1. Data Loading & Sanitization
file_17 = "cic2017_training_av_lbl_100K_hdr.csv"
file_18 = "cic2018_training_av_lbl_100K_hdr.csv"

def sanitize_data(df):
    df.columns = df.columns.astype(str)
    y = df['Label'].apply(lambda x: 0 if str(x).upper() == 'BENIGN' else 1)
    X = df.drop(columns=['Label'])
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(lambda x: x.fillna(x.max() if not np.isnan(x.max()) else 0), axis=0)
    return X, y

print("Loading and Sanitizing...")
X17, y17 = sanitize_data(pd.read_csv(file_17))
X18, y18 = sanitize_data(pd.read_csv(file_18))

# Evaluation set: Tail of 2018
test_X, test_y = X18.tail(5000), y18.tail(5000)

# 2. Train the 2017 Legacy Expert
print("Training Legacy Expert...")
expert = XGBClassifier(n_estimators=100, random_state=42)
expert.fit(X17, y17)

# 3. Execution Sweep
n_points = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
results = {'TL_Fixed': [], 'TL_Adaptive': [], 'XGB': [], 'MLP': []}
test_probs = expert.predict_proba(test_X)[:, 1]

print("Starting sweep...")
for n in n_points:
    train_X, train_y = X18.head(n), y18.head(n)
    
    # --- Local MLP (Non-Pretrained) ---
    if n < 2 or len(np.unique(train_y)) < 2:
        results['MLP'].append(0.0)
    else:
        mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)
        mlp.fit(train_X, train_y)
        results['MLP'].append(f1_score(test_y, mlp.predict(test_X)))

    # --- Local XGB (Non-Pretrained) ---
    if n < 2 or len(np.unique(train_y)) < 2:
        results['XGB'].append(0.0)
    else:
        xgb = XGBClassifier(n_estimators=50, random_state=42)
        xgb.fit(train_X, train_y)
        results['XGB'].append(f1_score(test_y, xgb.predict(test_X)))

    # --- TL Fixed (Source: 2017, Threshold: 0.08) ---
    results['TL_Fixed'].append(f1_score(test_y, (test_probs >= 0.08).astype(int)))

    # --- TL Adaptive (Practical Lagged Logic) ---
    if n == 0:
        thresh = 0.08 # Default to optimized baseline
    else:
        # Calibrate threshold based on the n samples seen so far
        recent_probs = expert.predict_proba(train_X)[:, 1]
        # We use the 95th percentile as the adaptive trigger
        thresh = np.percentile(recent_probs, 95) if len(recent_probs) > 0 else 0.08
    results['TL_Adaptive'].append(f1_score(test_y, (test_probs >= thresh).astype(int)))

# 4. Visualization
plt.figure(figsize=(12, 7))

# Plotting with distinct styles for clarity
plt.plot(n_points, results['TL_Adaptive'], 'o-', label='Adaptive TL (Dynamic Threshold)', color='#27ae60', lw=4)
plt.plot(n_points, results['TL_Fixed'], '--', label='Fixed TL (Threshold: 0.08)', color='#e67e22', lw=2)
plt.plot(n_points, results['XGB'], 's-', label='Local XGB (2018)', color='#34495e', lw=2)
plt.plot(n_points, results['MLP'], 'd--', label='Local MLP (2018)', color='#95a5a6', lw=2, alpha=0.7)

plt.title("Security Advantage: Multi-Model Sample Efficiency Comparison", fontsize=14, fontweight='bold')
plt.xlabel("Number of Labeled 2018 Samples (n)", fontsize=12)
plt.ylabel("F1-Score", fontsize=12)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.axvline(x=50, color='red', linestyle=':', alpha=0.5, label='Cold Start Horizon')

# Save and Show
plt.tight_layout()
plt.savefig("transfer_efficiency_full_comparison.png", dpi=300)
print("Plot saved: transfer_efficiency_full_comparison.png")
plt.show()