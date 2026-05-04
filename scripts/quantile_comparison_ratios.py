import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# 1. Setup Architecture & Load Model
class IntrusionDetector(nn.Module):
    def __init__(self, input_dim):
        super(IntrusionDetector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

df_18 = pd.read_csv('robust_2018_final.csv')
model = IntrusionDetector(df_18.shape[1] - 1)
model.load_state_dict(torch.load('ids_model.pth'))
model.eval()

benign_df = df_18[df_18['Label'] == 0]
attack_df = df_18[df_18['Label'] == 1]

# 2. Define the Test Parameters
ratios = np.linspace(10, 1, 10)
quantiles_to_test = [0.95, 0.75]
comparison_results = {q: [] for q in quantiles_to_test}

for ratio in ratios:
    num_attacks = 2000
    num_benign = int(num_attacks * ratio)
    test_set = pd.concat([benign_df.sample(num_benign), attack_df.sample(num_attacks)]).sample(frac=1)
    X_test = torch.tensor(test_set.drop(columns=['Label']).values, dtype=torch.float32)
    y_test = test_set['Label'].values

    with torch.no_grad():
        probs = model(X_test).squeeze().numpy()

    for q in quantiles_to_test:
        window_size = 50
        preds = []
        for i in range(len(probs)):
            win = probs[max(0, i-window_size):i+1]
            thresh = np.quantile(win, q)
            preds.append(1 if probs[i] >= thresh else 0)
        
        comparison_results[q].append(f1_score(y_test, preds))

# 3. Visualization
plt.figure(figsize=(10, 6))
for q, f1s in comparison_results.items():
    plt.plot(ratios, f1s, marker='o', label=f'Quantile {q}')

plt.gca().invert_xaxis()
plt.title("F1-Score: 0.95 vs 0.75 Quantile Sensitivity", fontsize=14)
plt.xlabel("Ratio (Benign : Attack)")
plt.ylabel("F1-Score")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('quantile_comparison_ratios.png', dpi=300)
plt.show()