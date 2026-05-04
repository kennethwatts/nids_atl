import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse
import random

# SET GLOBAL SEEDS
SEED = 99
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Ensure XGBoost also respects the seed
# (In your xgb.XGBClassifier, make sure random_state=SEED)
# 1. SETUP
parser = argparse.ArgumentParser(description='MLP vs XGBoost Ratio Comparison')
parser.add_argument('train_file', help='Path to 2017 dataset (CSV)')
parser.add_argument('test_file', help='Path to 2018 dataset (CSV)')
args = parser.parse_args()

# 2. MLP ARCHITECTURE
class IntrusionDetector(nn.Module):
    def __init__(self, input_dim):
        super(IntrusionDetector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# 3. DATA LOADING
def load_clean_data(path):
    print(f"Reading {path}...")
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df['Label'].astype(int).values
    X = df.drop(columns=['Label']).values
    return X, y

X_train_raw, y_train = load_clean_data(args.train_file)
X_test_raw, y_test = load_clean_data(args.test_file)

# IMPORTANT: Fit scaler ONCE on 2017 data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# 4. TRAINING
print("\nTraining Models...")
mlp_model = IntrusionDetector(X_train_scaled.shape[1])
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)
criterion = nn.BCELoss()

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

mlp_model.train()
for epoch in range(5):
    optimizer.zero_grad()
    loss = criterion(mlp_model(X_train_t), y_train_t)
    loss.backward()
    optimizer.step()

xgb_model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# 5. RATIO-AWARE PREDICTION FUNCTION
def get_adaptive_preds(probs, ratio):
    preds = []
    window = 100 # Increased window size for better stability in shuffled data
    
    # Calculate quantile based on current ratio: ratio/(ratio+1) is the benign fraction
    # We subtract a 5% buffer to ensure we catch the start of the attack hump
    benign_fraction = ratio / (ratio + 1)
    q = max(0.1, benign_fraction - 0.05) 
    
    for i in range(len(probs)):
        win_data = probs[max(0, i-window):i+1]
        thresh = np.quantile(win_data, q)
        preds.append(1 if probs[i] >= thresh else 0)
    return preds

# 6. EXPERIMENT LOOP
ratios = np.arange(1, 11)
results_mlp = []
results_xgb = []

benign_idx = np.where(y_test == 0)[0]
attack_idx = np.where(y_test == 1)[0]

for r in ratios:
    n_attack = 2000
    n_benign = int(n_attack * r)
    
    # Sample and Shuffle
    s_benign = np.random.choice(benign_idx, min(n_benign, len(benign_idx)), replace=False)
    s_attack = np.random.choice(attack_idx, min(n_attack, len(attack_idx)), replace=False)
    indices = np.concatenate([s_benign, s_attack])
    np.random.shuffle(indices)
    
    X_sample = X_test_scaled[indices]
    y_sample = y_test[indices]

    # Get Raw Scores
    mlp_model.eval()
    with torch.no_grad():
        mlp_probs = mlp_model(torch.tensor(X_sample, dtype=torch.float32)).squeeze().numpy()
    xgb_probs = xgb_model.predict_proba(X_sample)[:, 1]

    # Calculate F1 with the new ratio-aware logic
    results_mlp.append(f1_score(y_sample, get_adaptive_preds(mlp_probs, r)))
    results_xgb.append(f1_score(y_sample, get_adaptive_preds(xgb_probs, r)))
    
    print(f"Ratio 1:{r} | MLP F1: {results_mlp[-1]:.4f} | XGB F1: {results_xgb[-1]:.4f} | Threshold Q: {max(0.1, (r/(r+1))-0.05):.2f}")

# 7. VISUALIZATION
plt.figure(figsize=(10, 6))
plt.plot(ratios, results_mlp, marker='o', label='MLP (Neural Net)', color='#3498db', linewidth=2)
plt.plot(ratios, results_xgb, marker='s', label='XGBoost', color='#2ecc71', linewidth=2)
plt.title("MLP vs XGBoost: Ratio-Aware Adaptive Thresholding", fontsize=14)
plt.xlabel("Ratio (1:N Benign Packets)", fontsize=12)
plt.ylabel("F1-Score", fontsize=12)
plt.ylim(0, 1.1)
plt.legend()
plt.grid(alpha=0.3, linestyle='--')
plt.savefig('mlp_vs_xgboost_ratio_aware.png', dpi=300)
plt.show()