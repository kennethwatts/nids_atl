import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from numpy import trapz 
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 1. Load the specific CIC 100K datasets
file_17 = "cic2017_training_av_lbl_100K_hdr.csv"
file_18 = "cic2018_training_av_lbl_100K_hdr.csv"

print("Loading and Sanitizing datasets...")
df_17 = pd.read_csv(file_17)
df_18 = pd.read_csv(file_18)

def sanitize_data(df):
    df.columns = df.columns.astype(str)
    y = df['Label'].apply(lambda x: 0 if str(x).upper() == 'BENIGN' else 1)
    X = df.drop(columns=['Label'])
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(lambda x: x.fillna(x.max() if not np.isnan(x.max()) else 0), axis=0)
    return X, y

X17, y17 = sanitize_data(df_17)
X18, y18 = sanitize_data(df_18)

# 2. Setup Evaluation Set (last 5000 of 2018)
test_X = X18.tail(5000) 
test_y = y18.tail(5000)

# 3. Train the "Legacy Expert"
print("Training Legacy Expert on 2017 data...")
expert = XGBClassifier(n_estimators=100, random_state=42)
expert.fit(X17, y17)

# 4. The Sweep
n_points = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
results = {'TL_Fixed': [], 'TL_Adaptive': [], 'XGB': [], 'MLP': []}

print("Starting sweep of n = 0 to 100...")
# Get probabilities once to save time
test_probs = expert.predict_proba(test_X)[:, 1]

for n in n_points:
    train_X = X18.head(n)
    train_y = y18.head(n)
    
    # --- Local MLP & XGB ---
    if n < 2 or len(np.unique(train_y)) < 2:
        results['MLP'].append(0.0)
        results['XGB'].append(0.0)
    else:
        # MLP
        mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)
        mlp.fit(train_X, train_y)
        results['MLP'].append(f1_score(test_y, mlp.predict(test_X)))
        # XGB
        xgb = XGBClassifier(n_estimators=50, random_state=42)
        xgb.fit(train_X, train_y)
        results['XGB'].append(f1_score(test_y, xgb.predict(test_X)))
    
    # --- TL Fixed (0.08) ---
    results['TL_Fixed'].append(f1_score(test_y, (test_probs >= 0.08).astype(int)))
    
    # --- TL Adaptive (Practical Lagged Logic) ---
    # We simulate the threshold adaptation. As 'n' increases, the adaptive 
    # threshold moves from the 0.08 baseline toward the 95th percentile of new data.
    if n == 0:
        thresh = 0.08
    else:
        # Calculate the 95th percentile of the probabilities for the current 'n' samples
        recent_probs = expert.predict_proba(train_X)[:, 1]
        thresh = np.percentile(recent_probs, 95) if len(recent_probs) > 0 else 0.08
    
    results['TL_Adaptive'].append(f1_score(test_y, (test_probs >= thresh).astype(int)))

# 5. Calculate Gains and Plot
plt.figure(figsize=(12, 7))

plt.plot(n_points, results['TL_Adaptive'], 'o-', label='Adaptive TL (Dynamic Threshold)', color='#27ae60', lw=4)
plt.plot(n_points, results['TL_Fixed'], '--', label='Legacy TL (Fixed 0.08)', color='#e67e22', lw=2, alpha=0.7)
plt.plot(n_points, results['XGB'], 's-', label='Local XGB (2018 Only)', color='#34495e', lw=2)
plt.plot(n_points, results['MLP'], 'd--', label='Local MLP (2018 Only)', color='#95a5a6', lw=2)

plt.title("Sample Efficiency: Adaptive Transfer Learning vs. Baselines", fontsize=14, fontweight='bold')
plt.xlabel("Number of Labeled 2018 Samples (n)")
plt.ylabel("F1-Score")
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.axvline(x=50, color='red', linestyle=':', label='Cold Start Boundary')

plt.tight_layout()
plt.savefig("complete_transfer_efficiency_comparison.png", dpi=300)
print("Plot saved: complete_transfer_efficiency_comparison.png")
plt.show()