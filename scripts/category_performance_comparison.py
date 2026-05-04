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
parser = argparse.ArgumentParser(description='Categorized IDS Transfer Learning with Scaling')
parser.add_argument('train_file', help='Path to 2017 dataset (CSV)')
parser.add_argument('test_file', help='Path to 2018 dataset (CSV)')
args = parser.parse_args()

# 2. DEFINE MAPPING LOGIC
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

# 4. DATA PROCESSING & CLEANING
def load_and_clean(path):
    print(f"Reading {path}...")
    df = pd.read_csv(path)
    
    # Handle Infinity and NaN values (Common in CIC-IDS)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Map raw labels to categories
    df['Category'] = df['Label'].map(attack_map).fillna('BENIGN')
    df['BinaryLabel'] = (df['Category'] != 'BENIGN').astype(int)
    
    return df

print("Loading datasets...")
df_train = load_and_clean(args.train_file)
df_test = load_and_clean(args.test_file)

# Separate features
X_train_raw = df_train.drop(columns=['Label', 'Category', 'BinaryLabel']).values
y_train_raw = df_train['BinaryLabel'].values
X_test_raw = df_test.drop(columns=['Label', 'Category', 'BinaryLabel']).values

# 5. FEATURE SCALING (Crucial to prevent BCE Range Errors)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Convert to Tensors
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train_raw.reshape(-1, 1), dtype=torch.float32)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)

# 6. TRAINING ON 2017
model = IntrusionDetector(X_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
model.train()
print("Training model on scaled 2017 data...")
for epoch in range(3):
    total_loss = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        # The output of Sigmoid is now safely between 0 and 1
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")

# 7. CATEGORIZED EVALUATION ON 2018
model.eval()
categories = [c for c in df_test['Category'].unique() if c != 'BENIGN']
category_f1s = {}

print("\nEvaluating individual attack categories on 2018 data...")
for cat in categories:
    # Get indices for Benign and the specific Category
    mask = (df_test['Category'] == 'BENIGN') | (df_test['Category'] == cat)
    X_cat = X_test[mask]
    y_cat = df_test['BinaryLabel'].values[mask]

    with torch.no_grad():
        probs = model(X_cat).squeeze().numpy()

    # Adaptive Thresholding (Window=50, 95th Percentile)
    preds = []
    window_size = 50
    for i in range(len(probs)):
        win = probs[max(0, i-window_size):i+1]
        thresh = np.quantile(win, 0.95)
        preds.append(1 if probs[i] >= thresh else 0)

    score = f1_score(y_cat, preds)
    category_f1s[cat] = score
    print(f"Category: {cat:<15} | F1: {score:.4f}")

# 8. VISUALIZATION
plt.figure(figsize=(10, 6))
bars = plt.bar(category_f1s.keys(), category_f1s.values(), color='#34495e')
plt.title("Adaptive Threshold F1-Score by Category (Transfer 2017 -> 2018)", fontsize=14)
plt.ylabel("F1-Score")
plt.xlabel("Attack Category")
plt.ylim(0, 1.1)
plt.grid(axis='y', alpha=0.3)

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 3), ha='center')

plt.tight_layout()
plt.savefig('category_performance_normalized.png')
print("\nResults visualization saved as 'category_performance_normalized.png'")
plt.show()