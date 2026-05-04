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
    # Convert all column names to strings
    df.columns = df.columns.astype(str)
    # Separate features and labels
    y = df['Label'].apply(lambda x: 0 if str(x).upper() == 'BENIGN' else 1)
    X = df.drop(columns=['Label'])
    
    # Replace Infinity with NaN, then replace NaN with Column Max/0
    X = X.replace([np.inf, -np.inf], np.nan)
    # For each column, fill NaN with the maximum finite value found in that column
    X = X.apply(lambda x: x.fillna(x.max() if not np.isnan(x.max()) else 0), axis=0)
    
    return X, y

X17, y17 = sanitize_data(df_17)
X18, y18 = sanitize_data(df_18)

# 2. Setup fixed Evaluation Set (last 5000 of 2018)
test_X = X18.tail(5000) 
test_y = y18.tail(5000)

# 3. Train the "Legacy Expert"
print("Training Legacy Expert on 2017 data...")
expert = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False)
expert.fit(X17, y17)

# 4. The Sample Efficiency Sweep
n_points = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
results = {'TL': [], 'XGB': [], 'MLP': []}

print("Starting sweep of n = 0 to 100...")
for n in n_points:
    train_X = X18.head(n)
    train_y = y18.head(n)
    
    # --- Local MLP ---
    if n < 2 or len(np.unique(train_y)) < 2:
        mlp_f1 = 0.0
    else:
        local_mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)
        try:
            local_mlp.fit(train_X, train_y)
            mlp_f1 = f1_score(test_y, local_mlp.predict(test_X))
        except:
            mlp_f1 = 0.0
    results['MLP'].append(mlp_f1)
    
    # --- Local XGB ---
    if n < 2 or len(np.unique(train_y)) < 2:
        xgb_f1 = 0.0
    else:
        local_xgb = XGBClassifier(n_estimators=50, random_state=42)
        local_xgb.fit(train_X, train_y)
        xgb_f1 = f1_score(test_y, local_xgb.predict(test_X))
    results['XGB'].append(xgb_f1)
    
    # --- TL Path ---
    probs = expert.predict_proba(test_X)[:, 1]
    tl_f1 = f1_score(test_y, (probs >= 0.08).astype(int))
    results['TL'].append(tl_f1)

# 5. Calculate Gains and Plot
auc_mlp = trapz(results['MLP'][:6], n_points[:6])
auc_tl = trapz(results['TL'][:6], n_points[:6])
gain = auc_tl / max(auc_mlp, 0.001)

plt.figure(figsize=(10, 6))
plt.plot(n_points, results['TL'], 'o-', label='Transfer Learning (Source: 2017)', color='#e67e22', lw=3)
plt.plot(n_points, results['XGB'], 's-', label='Local XGB (2018 Only)', color='#34495e', lw=2)
plt.plot(n_points, results['MLP'], 'd--', label='Local MLP (2018 Only)', color='#95a5a6', lw=2, alpha=0.7)

plt.title(f"Sample Efficiency (TL Gain vs MLP: {gain:.2f}x)", fontsize=14, fontweight='bold')
plt.xlabel("Number of Labeled 2018 Samples (n)")
plt.ylabel("F1-Score")
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.axvline(x=50, color='red', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig("final_transfer_gain_comparison_mlp.png", dpi=300)
print(f"Plot saved to: final_transfer_gain_comparison_mlp.png")
plt.show()