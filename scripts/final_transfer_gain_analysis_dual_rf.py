import numpy as np
import matplotlib.pyplot as plt

# 1. DATA FROM RF VERSION RESULTS
labels = [0, 10, 20, 50, 100]
res_local_rf = [0.000, 0.287, 0.382, 0.602, 0.750]
res_mlp = [0.000, 0.320, 0.419, 0.624, 0.768]
res_tl = [0.551, 0.560, 0.567, 0.635, 0.769]

# 2. AREA CALCULATIONS (Cold Start: n=0 to n=50)
x_cold = np.array(labels[:4])
area_tl_cold = np.trapz(res_tl[:4], x_cold)
area_rf_cold = np.trapz(res_local_rf[:4], x_cold)
area_mlp_cold = np.trapz(res_mlp[:4], x_cold)

# Gain Calculation
gain_rf = area_tl_cold / area_rf_cold

# 3. VISUALIZATION
plt.figure(figsize=(12, 7))

# Main Lines
plt.plot(labels, res_local_rf, label='Local Random Forest (2018)', color='#e67e22', marker='s', linestyle='--')
plt.plot(labels, res_mlp, label='Local MLP (2018)', color='#95a5a6', marker='d', linestyle='--')
plt.plot(labels, res_tl, label='Adaptive TL (Path Transfer)', color='#2980b9', marker='^', linewidth=3)

# Shading Phase 1: Cold Start Advantage
plt.fill_between(labels[:4], res_local_rf[:4], res_tl[:4], 
                 where=(np.array(res_tl[:4]) >= np.array(res_local_rf[:4])), 
                 color='green', alpha=0.15, hatch='//', 
                 label=f'Cold Start Gain (vs RF): {gain_rf:.2f}x')

# Shading Phase 2: Maturation Zone
plt.fill_between(labels[3:], res_local_rf[3:], res_tl[3:], color='gray', alpha=0.05, label='Convergence Zone')

# Structural Annotations
plt.axvline(x=50, color='#c0392b', linestyle='-.', alpha=0.6)
plt.text(8, 0.70, "PHASE 1: COLD START\n(Transfer Advantage)", fontsize=10, fontweight='bold', color='#2c3e50')
plt.text(60, 0.70, "PHASE 2: MATURATION\n(Model Convergence)", fontsize=10, fontweight='bold', color='#7f8c8d')

# Professional Formatting
plt.title("Security Advantage & Path Transfer Gain (Random Forest Core)", fontsize=15, fontweight='bold')
plt.xlabel("Number of Labeled 2018 Samples ($n$)", fontsize=12)
plt.ylabel("Detection Efficacy (F1-Score)", fontsize=12)
plt.legend(loc='lower right', frameon=True, shadow=True, fontsize=10)
plt.grid(alpha=0.3, linestyle='--')
plt.ylim(0, 0.85)
plt.tight_layout()

plt.savefig('dual_phase_transfer_analysis_rf.png', dpi=300)
plt.show()