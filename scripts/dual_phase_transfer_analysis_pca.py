import numpy as np
import matplotlib.pyplot as plt

# 1. DATA FROM PCA VERSION RESULTS
labels = [0, 10, 20, 50, 100]
res_local_xgb = [0.000, 0.059, 0.294, 0.582, 0.765]
res_mlp = [0.000, 0.247, 0.341, 0.423, 0.460]
res_tl = [0.378, 0.408, 0.493, 0.628, 0.768]

# 2. AREA CALCULATIONS (Cold Start: n=0 to n=50)
x_cold = np.array(labels[:4])
area_tl_cold = np.trapz(res_tl[:4], x_cold)
area_xgb_cold = np.trapz(res_local_xgb[:4], x_cold)

# Gain Calculation
gain_xgb = area_tl_cold / area_xgb_cold

# 3. VISUALIZATION
plt.figure(figsize=(12, 7))

# Main Lines
plt.plot(labels, res_local_xgb, label='Local XGB (PCA Space)', color='#e74c3c', marker='o', linestyle='--')
plt.plot(labels, res_mlp, label='Local MLP (PCA Space)', color='#95a5a6', marker='d', linestyle='--')
plt.plot(labels, res_tl, label='Adaptive TL (Path Transfer)', color='#2980b9', marker='^', linewidth=3)

# Shading Phase 1: Cold Start Advantage
plt.fill_between(labels[:4], res_local_xgb[:4], res_tl[:4], 
                 where=(np.array(res_tl[:4]) >= np.array(res_local_xgb[:4])), 
                 color='green', alpha=0.15, hatch='//', 
                 label=f'Cold Start Gain (vs XGB): {gain_xgb:.2f}x')

# Shading Phase 2: Maturation Zone
plt.fill_between(labels[3:], res_local_xgb[3:], res_tl[3:], color='gray', alpha=0.05, label='Convergence Zone')

# Structural Annotations
plt.axvline(x=50, color='#c0392b', linestyle='-.', alpha=0.6)
plt.text(8, 0.70, "PHASE 1: COLD START\n(Transfer Advantage)", fontsize=10, fontweight='bold', color='#2c3e50')
plt.text(60, 0.70, "PHASE 2: MATURATION\n(Model Convergence)", fontsize=10, fontweight='bold', color='#7f8c8d')

# Professional Formatting
plt.title("Security Advantage: Path Transfer Gain in PCA Latent Space", fontsize=15, fontweight='bold')
plt.xlabel("Number of Labeled 2018 Samples ($n$)", fontsize=12)
plt.ylabel("Detection Efficacy (F1-Score)", fontsize=12)
plt.legend(loc='lower right', frameon=True, shadow=True, fontsize=10)
plt.grid(alpha=0.3, linestyle='--')
plt.ylim(0, 0.85)
plt.tight_layout()

plt.savefig('dual_phase_transfer_analysis_pca.png', dpi=300)
plt.show()