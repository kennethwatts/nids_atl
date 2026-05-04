import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# 1. ARCHITECTURE DEFINITION
class IntrusionDetector(nn.Module):
    def __init__(self, input_dim):
        super(IntrusionDetector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# 2. DATA LOADING & MODEL PREPARATION
# Note: Ensure 'robust_2018_final.csv' and 'ids_model.pth' are in your working directory
df_18 = pd.read_csv('robust_2018_final.csv')
input_dim = df_18.shape[1] - 1
model = IntrusionDetector(input_dim)
model.load_state_dict(torch.load('ids_model.pth'))
model.eval()

# Separate classes to control the ratios
benign_df = df_18[df_18['Label'] == 0]
attack_df = df_18[df_18['Label'] == 1]

# 3. SENSITIVITY ANALYSIS PARAMETERS
ratios = np.linspace(10, 1, 10) 
quantiles_to_test = [0.95, 0.75]
comparison_results = {q: [] for q in quantiles_to_test}

print(f"{'Ratio':<10} | {'Attack %':<10} | {'F1 (0.95)':<10} | {'F1 (0.75)':<10}")
print("-" * 55)

for ratio in ratios:
    # Sample data for this specific ratio
    num_attacks = 2000 
    num_benign = int(num_attacks * ratio)
    
    s_benign = benign_df.sample(min(len(benign_df), num_benign), random_state=42)
    s_attack = attack_df.sample(min(len(attack_df), num_attacks), random_state=42)
    
    test_set = pd.concat([s_benign, s_attack]).sample(frac=1, random_state=42)
    X_test = torch.tensor(test_set.drop(columns=['Label']).values, dtype=torch.float32)
    y_test = test_set['Label'].values

    with torch.no_grad():
        probs = model(X_test).squeeze().numpy()

    # Calculate and store F1 scores
    current_f1s = {}
    for q in quantiles_to_test:
        window_size = 50
        preds = []
        for i in range(len(probs)):
            start = max(0, i - window_size)
            window = probs[start:i+1]
            thresh = np.quantile(window, q)
            preds.append(1 if probs[i] >= thresh else 0)
        
        score = f1_score(y_test, preds)
        comparison_results[q].append(score)
        current_f1s[q] = score
    
    # Console Feedback
    attack_pct = 1 / (ratio + 1)
    print(f"{int(ratio):>2}:1      | {attack_pct:>8.1%} | {current_f1s[0.95]:>10.4f} | {current_f1s[0.75]:>10.4f}")
    
    # 4. CLEANED DISTRIBUTION HISTOGRAM (Triggered at 1:1 ratio)
    if ratio == 1.0:
        plt.figure(figsize=(10, 6))
        # Plot distributions with standardized range for better comparison
        plt.hist(probs[y_test==0], bins=60, range=(0, 0.6), alpha=0.6, label='Benign Packets', color='#3498db')
        plt.hist(probs[y_test==1], bins=60, range=(0, 0.6), alpha=0.6, label='Attack Packets', color='#e74c3c')

        # Draw decision boundary lines
        q95_val = np.quantile(probs, 0.95)
        q75_val = np.quantile(probs, 0.75)
        
        plt.axvline(q95_val, color='red', linestyle='--', linewidth=2, label=f'95th Percentile Threshold ({q95_val:.3f})')
        plt.axvline(q75_val, color='green', linestyle='--', linewidth=2, label=f'75th Percentile Threshold ({q75_val:.3f})')

        plt.title("Statistical Overcrowding Analysis: 1:1 Benign-to-Attack Ratio", fontsize=14)
        plt.xlabel("Model Confidence (Sigmoid Probability Output)", fontsize=12)
        plt.ylabel("Packet Frequency", fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig('probability_distribution_1to1_clean.png', dpi=300)
        plt.show()

# 5. FINAL SENSITIVITY PLOT
plt.figure(figsize=(12, 6))
plt.plot(ratios, comparison_results[0.95], marker='o', label='Strict Mode (0.95 Quantile)', color='#e74c3c', linewidth=2)
plt.plot(ratios, comparison_results[0.75], marker='s', label='Balanced Mode (0.75 Quantile)', color='#3498db', linewidth=2)

# Orientation: 1:1 on the left, 10:1 on the right
plt.xticks(ratios, [f"{int(r)}:1" for r in ratios])
plt.title("Impact of Class Imbalance on Adaptive Threshold Performance", fontsize=14)
plt.xlabel("Class Ratio (Benign : Attack)", fontsize=12)
plt.ylabel("F1-Score", fontsize=12)
plt.grid(alpha=0.3, linestyle='--')
plt.legend()
plt.tight_layout()
plt.savefig('quantile_sensitivity_comparison_final.png', dpi=300)
plt.show()