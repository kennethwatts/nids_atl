import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from numpy import trapz 
import warnings

warnings.filterwarnings("ignore")

# 1. Load your 100K datasets
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

# Use a fixed test set from the end of 2018
test_X, test_y = X18.tail(5000), y18.tail(5000)

# 2. Train the 2017 Legacy Expert
print("Training Legacy Expert...")
expert = XGBClassifier(n_estimators=100, random_state=42)
expert.fit(X17, y17)

# 3. Sweep
n_points = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
results = {'TL': [], 'XGB': [], 'MLP': []}
test_probs = expert.predict_proba(test_X)[:, 1]

for n in n_points:
    train_X, train_y = X18.head(n), y18.head(n)
    
    # Local MLP (Non-Pretrained)
    if n < 2 or len(np.unique(train_y)) < 2:
        results['MLP'].append(0.0)
    else:
        mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)
        mlp.fit(train_X, train_y)
        results['MLP'].append(f1_score(test_y, mlp.predict(test_X)))

    # Local XGB (Non-Pretrained)
    if n < 2 or len(np.unique(train_y)) < 2:
        results['XGB'].append(0.0)
    else:
        xgb = XGBClassifier(n_estimators=50, random_state=42)
        xgb.fit(train_X, train_y)
        results['XGB'].append(f1_score(test_y, xgb.predict(test_X)))

    # Transfer Learning (Pretrained 2017 + Optimized 0.08)
    results['TL'].append(f1_score(test_y, (test_probs >= 0.08).astype(int)))

# 4. Visualization
plt.figure(figsize=(10, 6))
plt.plot(n_points, results['TL'], 'o-', label='Transfer Learning (Source: 2017)', color='#e67e22', lw=3)
plt.plot(n_points, results['XGB'], 's-', label='Local XGB (2018 Only)', color='#34495e', lw=2)
plt.plot(n_points, results['MLP'], 'd--', label='Local MLP (2018 Only)', color='#95a5a6', lw=2)

# Calculation of Transfer Gain vs MLP for the title
auc_mlp = trapz(results['MLP'][:6], n_points[:6])
auc_tl = trapz(results['TL'][:6], n_points[:6])
gain = auc_tl / max(auc_mlp, 0.001)

plt.title(f"Security Advantage: TL vs. Local Baselines (Gain vs MLP: {gain:.2f}x)")
plt.xlabel("Number of Labeled 2018 Samples (n)")
plt.ylabel("F1-Score")
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig("final_transfer_gain_with_mlp.png", dpi=300)
plt.show()