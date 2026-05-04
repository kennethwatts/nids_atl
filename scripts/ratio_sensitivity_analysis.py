import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# 1. Setup Architecture (Same as before)
class IntrusionDetector(nn.Module):
    def __init__(self, input_dim):
        super(IntrusionDetector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# 2. Load Data and Model
df_18 = pd.read_csv('robust_2018_final.csv')
input_dim = df_18.shape[1] - 1
model = IntrusionDetector(input_dim)
model.load_state_dict(torch.load('ids_model.pth'))
model.eval()

# Separate Benign and Attack for easier sampling
benign_df = df_18[df_18['Label'] == 0]
attack_df = df_18[df_18['Label'] == 1]

# 3. Define the Ratio Steps
# From 10:1 (Benign:Attack) down to 1:1
ratios = np.linspace(10, 1, 10) # [10.0, 9.0, ..., 1.0]
results = []

print("--- Running Ratio Sensitivity Analysis ---")

for ratio in ratios:
    # Fix the number of attacks and sample benign based on ratio
    num_attacks = 2000 
    num_benign = int(num_attacks * ratio)
    
    # Ensure we don't exceed available data
    sampled_benign = benign_df.sample(min(len(benign_df), num_benign), random_state=42)
    sampled_attack = attack_df.sample(min(len(attack_df), num_attacks), random_state=42)
    
    test_set = pd.concat([sampled_benign, sampled_attack]).sample(frac=1) # Combine and shuffle
    X_test = torch.tensor(test_set.drop(columns=['Label']).values, dtype=torch.float32)
    y_test = test_set['Label'].values

    # Predict
    with torch.no_grad():
        probs = model(X_test).squeeze().numpy()

    # Adaptive Threshold (Window=50, 95th Percentile)
    window_size = 50
    adaptive_preds = []
    for i in range(len(probs)):
        start = max(0, i - window_size)
        window = probs[start:i+1]
        thresh = np.quantile(window, 0.95)
        adaptive_preds.append(1 if probs[i] >= thresh else 0)

    f1 = f1_score(y_test, adaptive_preds)
    results.append({'Ratio': f"{int(ratio)}:1", 'F1': f1, 'Attack_Pct': 1/(ratio+1)})
    print(f"Ratio {int(ratio)}:1 (Attack {1/(ratio+1):.1%}) -> F1: {f1:.4f}")

# 4. Visualization & Saving
res_df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
plt.plot(res_df['Ratio'], res_df['F1'], marker='o', linestyle='-', color='#2ecc71', linewidth=2)
plt.gca().invert_xaxis() # Show 10:1 to 1:1 progression
plt.title("F1-Score Sensitivity to Benign:Attack Ratio", fontsize=14)
plt.xlabel("Ratio (Benign : Attack)", fontsize=12)
plt.ylabel("F1-Score", fontsize=12)
plt.grid(alpha=0.3)

# Add labels
for i, txt in enumerate(res_df['F1']):
    plt.annotate(f"{txt:.2f}", (res_df['Ratio'][i], res_df['F1'][i]), 
                 textcoords="offset points", xytext=(0,10), ha='center')

plt.savefig('ratio_sensitivity_analysis.png', dpi=300)
plt.show()