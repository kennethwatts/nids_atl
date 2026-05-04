import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse

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

# 3. DATA LOADING & PREP
def load_clean_data(path):
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna(0)
    X = df.drop(columns=['Label']).values
    y = (df['Label'] != 'BENIGN').astype(int).values
    return X, y, df

print("Loading data and fitting scaler...")
X_train_raw, y_train, _ = load_clean_data(args.train_file)
X_test_raw, y_test, df_test = load_clean_data(args.test_file)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# 4. TRAIN BOTH MODELS
print("Training MLP (Neural Network)...")
input_dim = X_train_scaled.shape[1]
mlp_model = IntrusionDetector(input_dim)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

mlp_model.train()
for epoch in range(3):
    optimizer.zero_grad()
    loss = criterion(mlp_model(X_train_t), y_train_t)
    loss.backward()
    optimizer.step()

print("Training XGBoost...")
xgb_model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# 5. RATIO EXPERIMENT
ratios = np.linspace(1, 10, 10) # 1:1 to 1:10
results_mlp = []
results_xgb = []

benign_idx = np.where(y_test == 0)[0]
attack_idx = np.where(y_test == 1)[0]

print("\nStarting Ratio Comparison (using 0.75 quantile)...")

for r in ratios:
    # Sample based on ratio
    n_attack = 2000
    n_benign = int(n_attack * r)
    
    selected_benign = np.random.choice(benign_idx, n_benign, replace=False)
    selected_attack = np.random.choice(attack_idx, n_attack, replace=False)
    
    indices = np.concatenate([selected_benign, selected_attack])
    np.random.shuffle(indices)
    
    X_sample = X_test_scaled[indices]
    y_sample = y_test[indices]

    # Get Probabilities
    mlp_model.eval()
    with torch.no_grad():
        mlp_probs = mlp_model(torch.tensor(X_sample, dtype=torch.float32)).squeeze().numpy()
    xgb_probs = xgb_model.predict_proba(X_sample)[:, 1]

    # Adaptive Thresholding (0.75 Quantile)
    def get_preds(probs, q=0.75):
        preds = []
        window = 50
        for i in range(len(probs)):
            win_data = probs[max(0, i-window):i+1]
            thresh = np.quantile(win_data, q)
            preds.append(1 if probs[i] >= thresh else 0)
        return preds

    results_mlp.append(f1_score(y_sample, get_preds(mlp_probs)))
    results_xgb.append(f1_score(y_sample, get_preds(xgb_probs)))
    print(f"Ratio 1:{int(r)} - MLP: {results_mlp[-1]:.4f}, XGB: {results_xgb[-1]:.4f}")

# 6. VISUALIZATION
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

# MLP Plot
ax1.plot(ratios, results_mlp, marker='o', color='#3498db', linewidth=2)
ax1.set_title("MLP Performance (0.75 Quantile)")
ax1.set_xlabel("Ratio (1:N Benign)")
ax1.set_ylabel("F1-Score")
ax1.grid(alpha=0.3)

# XGBoost Plot
ax2.plot(ratios, results_xgb, marker='s', color='#27ae60', linewidth=2)
ax2.set_title("XGBoost Performance (0.75 Quantile)")
ax2.set_xlabel("Ratio (1:N Benign)")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('mlp_vs_xgboost_ratios.png', dpi=300)
print("\nComparison graph saved as 'mlp_vs_xgboost_ratios.png'")
plt.show()