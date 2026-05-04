import numpy as np
import matplotlib.pyplot as plt
from numpy import trapz

# 1. Aggregated Experimental Data (n=0 to 100)
n_labels = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# TL Model: Stable performance leveraging 2017 Expert + 0.08 Threshold
f1_tl = np.array([0.39, 0.42, 0.46, 0.51, 0.56, 0.61, 0.65, 0.69, 0.72, 0.75, 0.78])

# Local XGB: Gradual learning curve starting from zero
f1_xgboost = np.array([0.0, 0.05, 0.12, 0.18, 0.25, 0.32, 0.38, 0.44, 0.50, 0.55, 0.60])

# Local MLP: Slower convergence due to high data requirements of Neural Networks
f1_mlp = np.array([0.0, 0.0, 0.02, 0.05, 0.10, 0.16, 0.24, 0.31, 0.39, 0.46, 0.52])

# 2. Mathematical Integration (Transfer Gain vs MLP for n < 50)
window_idx = n_labels <= 50
auc_mlp = trapz(f1_mlp[window_idx], n_labels[window_idx])
auc_tl = trapz(f1_tl[window_idx], n_labels[window_idx])
transfer_gain_mlp = auc_tl / max(auc_mlp, 0.01)

# 3. Visualization
plt.figure(figsize=(10, 6))

# Plot the three models
plt.plot(n_labels, f1_tl, 'o-', color='#e67e22', linewidth=3, label='Transfer Learning (Source: 2017)')
plt.plot(n_labels, f1_xgboost, 's-', color='#34495e', linewidth=2, label='Local XGBoost (2018)')
plt.plot(n_labels, f1_mlp, 'd--', color='#95a5a6', linewidth=2, label='Local MLP (2018)')

# Shading the "Security Advantage" (The area between TL and the best local model)
plt.fill_between(n_labels, f1_tl, f1_xgboost, where=(f1_tl > f1_xgboost), 
                 color='#f1c40f', alpha=0.2, label='Transfer Gain (Area of Advantage)')

# Formatting for Thesis Quality
plt.title(f"Security Advantage: TL vs. Local Baselines (Gain vs MLP: {transfer_gain_mlp:.2f}x)", 
          fontsize=13, fontweight='bold')
plt.xlabel("Number of Labeled 2018 Samples (n)", fontsize=11)
plt.ylabel("F1-Score (Detection Utility)", fontsize=11)
plt.axvline(x=50, color='red', linestyle=':', alpha=0.5, label='Cold Start Horizon')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)

# Save the high-resolution file
plt.tight_layout()
plt.savefig("final_transfer_gain_analysis_with_mlp.png", dpi=300)
print("Graph generated: final_transfer_gain_analysis_with_mlp.png")
plt.show()