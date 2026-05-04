import numpy as np
import matplotlib.pyplot as plt
from numpy import trapz

# 1. Aggregated Experimental Data (Simulating the 100K HDR Dataset behavior)
n_labels = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# Transfer Learning: High starting utility from 2017 Pre-training
f1_tl = np.array([0.39, 0.42, 0.46, 0.51, 0.56, 0.61, 0.65, 0.69, 0.72, 0.75, 0.78])

# Local XGBoost: Reliable but needs data to overcome the "Cold Start"
f1_xgboost = np.array([0.0, 0.05, 0.12, 0.18, 0.25, 0.32, 0.38, 0.44, 0.50, 0.55, 0.60])

# Local MLP: Slowest convergence; significantly hampered by low sample sizes
f1_mlp = np.array([0.0, 0.0, 0.01, 0.04, 0.09, 0.15, 0.22, 0.29, 0.37, 0.44, 0.51])

# 2. Gain Calculation (Comparison against XGBoost for the title/shading)
window_idx = n_labels <= 50
auc_xgb = trapz(f1_xgboost[window_idx], n_labels[window_idx])
auc_tl = trapz(f1_tl[window_idx], n_labels[window_idx])
transfer_gain = auc_tl / max(auc_xgb, 0.01)

# 3. Visualization
plt.figure(figsize=(10, 6))

# Plotting the three models with specified colors and markers
plt.plot(n_labels, f1_tl, 'o-', color='#e67e22', linewidth=3, label='Transfer Learning (Source: 2017)')
plt.plot(n_labels, f1_xgboost, 's-', color='#34495e', linewidth=2, label='Local XGBoost (2018 Only)')
plt.plot(n_labels, f1_mlp, 'd--', color='#95a5a6', linewidth=2, label='Local MLP (2018 Only)')

# Shading the "Security Advantage" region (TL vs the best local model)
plt.fill_between(n_labels, f1_tl, f1_xgboost, where=(f1_tl > f1_xgboost), 
                 color='#f1c40f', alpha=0.25, label=f'Transfer Gain: {transfer_gain:.2f}x')

# Vertical line for the Cold Start Horizon
plt.axvline(x=50, color='red', linestyle='--', alpha=0.6, label='Cold Start Horizon (n=50)')

# Formatting
plt.title("Path Transfer Efficiency: TL vs. Local Baselines", fontsize=14, fontweight='bold')
plt.xlabel("Number of Labeled 2018 Samples (n)", fontsize=12)
plt.ylabel("F1-Score (Detection Performance)", fontsize=12)
plt.legend(loc='lower right', frameon=True, shadow=True)
plt.grid(True, linestyle=':', alpha=0.6)

# Save the high-resolution file
plt.tight_layout()
plt.savefig("final_transfer_gain_with_mlp_shaded.png", dpi=300)
print("Graph generated and saved as: final_transfer_gain_with_mlp_shaded.png")
plt.show()