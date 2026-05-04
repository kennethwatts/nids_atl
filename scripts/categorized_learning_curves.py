import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse

# 1. SETUP COMMAND LINE ARGUMENTS
parser = argparse.ArgumentParser(description='Categorized Learning Curves (10%-100%)')
parser.add_argument('train_file', help='Path to 2017 dataset (CSV)')
parser.add_argument('test_file', help='Path to 2018 dataset (CSV)')
args = parser.parse_args()

# 2. UNIFIED BEHAVIORAL MAPPING
attack_map = {
    'BENIGN': 'BENIGN',
    'DDoS': 'Volumetric', 'DoS Hulk': 'Volumetric', 'DoS GoldenEye': 'Volumetric',
    'DoS Slowhttptest': 'Volumetric', 'DoS slowloris': 'Volumetric',
    'FTP-Patator': 'Authentication', 'SSH-Patator': 'Authentication', 'Web Attack - Brute Force': 'Authentication',
    'Web Attack - Sql Injection': 'Web/Injection', 'Web Attack - XSS': 'Web/Injection',
    'PortScan': 'Reconnaissance',
    'Bot': 'Persistence', 'Infiltration': 'Persistence'
}

# 3. ARCHITECTURE
class IntrusionDetector(nn.Module):
    def __init__(self, input_dim):
        super(IntrusionDetector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# 4. DATA CLEANING & LOADING
def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df['Category'] = df['Label'].map(attack_map).fillna('BENIGN')
    df['BinaryLabel'] = (df['Category'] != 'BENIGN').astype(int)
    return df

print("Loading datasets...")
df_train = load_and_clean(args.train_file)
df_test = load_and_clean(args.test_file)

# 5. PREPARE SCALER & TRAINING DATA (2017)
X_train_raw = df_train.drop(columns=['Label', 'Category', 'BinaryLabel']).values
y_train_raw = df_train['BinaryLabel'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_raw.reshape(-1, 1), dtype=torch.float32)

# 6. TRAIN BASE MODEL ON 2017
model = IntrusionDetector(X_train_tensor.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()
loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)

model.train()
print("Training base model on 2017 data...")
for epoch in range(3):
    for bx, by in loader:
        optimizer.zero_grad()
        criterion(model(bx), by).backward()
        optimizer.step()

# 7. EXPERIMENT: INCREMENTAL DATA (10% to 100%) PER CATEGORY
model.eval()
categories = [c for c in df_test['Category'].unique() if c != 'BENIGN']
percentages = np.linspace(0.1, 1.0, 10)
results = {cat: [] for cat in categories}

print("\nRunning incremental evaluation for each category...")

for pct in percentages:
    for cat in categories:
        # Filter for Benign + this Category
        cat_df = pd.concat([df_test[df_test['Category'] == 'BENIGN'], 
                            df_test[df_test['Category'] == cat]])
        
        # Sample the specific percentage
        sample_size = int(len(cat_df) * pct)
        cat_sample = cat_df.sample(n=sample_size, random_state=42)
        
        X_eval_raw = cat_sample.drop(columns=['Label', 'Category', 'BinaryLabel']).values
        y_eval = cat_sample['BinaryLabel'].values
        
        # Scale using the 2017 scaler
        X_eval_scaled = scaler.transform(X_eval_raw)
        X_eval_tensor = torch.tensor(X_eval_scaled, dtype=torch.float32)

        with torch.no_grad():
            probs = model(X_eval_tensor).squeeze().numpy()

        # Adaptive Threshold (Window=50, 95th Percentile)
        preds = []
        window_size = 50
        for i in range(len(probs)):
            win = probs[max(0, i-window_size):i+1]
            thresh = np.quantile(win, 0.95)
            preds.append(1 if probs[i] >= thresh else 0)

        results[cat].append(f1_score(y_eval, preds))
    print(f"Completed {int(pct*100)}% evaluation.")

# 8. VISUALIZATION
plt.figure(figsize=(12, 7))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6']

for i, cat in enumerate(categories):
    plt.plot(percentages * 100, results[cat], marker='o', label=cat, color=colors[i % len(colors)], linewidth=2)

plt.title("Categorized Transfer Learning: F1-Score vs. Data Percentage", fontsize=14)
plt.xlabel("Percentage of 2018 Data Used (%)", fontsize=12)
plt.ylabel("F1-Score", fontsize=12)
plt.ylim(0, 1.1)
plt.grid(alpha=0.3, linestyle='--')
plt.legend(title="Attack Category", loc='lower right')
plt.tight_layout()

plt.savefig('categorized_learning_curves.png', dpi=300)
print("\nFinal graph saved as 'categorized_learning_curves.png'")
plt.show()