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

# 2. LOAD DATA AND MODEL
df_18 = pd.read_csv('robust_2018_final.csv')
input_dim = df_18.shape[1] - 1
model = IntrusionDetector(input_dim)

# Load the saved 2017 intelligence
model.load_state_dict(torch.load('ids_model.pth'))
model.eval()

# Separate classes for sampling
benign_df = df_18[df_18['Label'] == 0]
attack_df = df_18[df_18['Label'] == 1]

# 3. DEFINE RATIOS AND QUANTILES
ratios = np.linspace(10, 1, 10)  # [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
quantiles_to_test = [0.95, 0.75]
comparison_results = {q: [] for q in quantiles_to_test}

print("--- Running Quantile Sensitivity Analysis ---")

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

    # Test both thresholds on the SAME sample
    for q in quantiles_to_test:
        window_size = 50
        preds = []
        for i in range(len(probs)):
            start = max(0, i - window_size)
            window = probs[start:i+1]
            thresh = np.quantile(window, q)
            preds.append(1 if probs[i] >= thresh else 0)
        
        f1 = f1_score(y_test, preds)
        comparison_results[q].append(f1)
    
    print(f"Ratio {int(ratio)}:1 processed.")

# 4. VISUALIZATION (1:1 ON THE LEFT, 10:1 ON THE RIGHT)
plt.figure(figsize=(12, 6))

# Plot both lines
plt.plot(ratios, comparison_results[0.95], marker='o', label='Strict Quantile (0.95)', 
         color='#e74c3c', linewidth=2)
plt.plot(ratios, comparison_results[0.75], marker='s', label='Balanced Quantile (0.75)', 
         color='#3498db', linewidth=2)

# REMOVED invert_xaxis() - By default, ratios [10...1] would plot 1 on left, 10 on right
# We ensure the ticks are clear
plt.xticks(ratios, [f"{int(r)}:1" for r in ratios])

plt.title("F1-Score Sensitivity: Quantile Threshold Comparison", fontsize=14)
plt.xlabel("Ratio (Benign : Attack)", fontsize=12)
plt.ylabel("F1-Score", fontsize=12)
plt.grid(alpha=0.3, linestyle='--')
plt.legend()

# Save the properly oriented graph
plt.savefig('quantile_ratio_comparison_final.png', dpi=300, bbox_inches='tight')
print("\nGraph saved as 'quantile_ratio_comparison_final.png'")
plt.show()