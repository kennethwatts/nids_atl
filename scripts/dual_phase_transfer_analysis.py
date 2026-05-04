import numpy as np
import matplotlib.pyplot as plt

# 1. FINAL AVERAGED EXPERIMENTAL DATA
labels = [0, 10, 20, 50, 100]
res_local_xgb = [0.000, 0.023, 0.271, 0.588, 0.761]
res_mlp = [0.000, 0.298, 0.439, 0.641, 0.766]
res_tl = [0.433, 0.447, 0.505, 0.640, 0.759]

# 2. AREA CALCULATIONS
# Cold Start Area (0 to 50)
x_cold = np.array(labels[:4])
area_tl_cold = np.trapz(res_tl[:4], x_cold)
area_xgb_cold = np.trapz(res_local_xgb[:4], x_cold)
area_mlp_cold = np.trapz(res_mlp[:4], x_cold)

# Total Area (0 to 100)
x_total = np.array(labels)
area_tl_total = np.trapz(res_tl, x_total)
area_xgb_total = np.trapz(res_local_xgb, x_total)
area_mlp_total = np.trapz(res_mlp, x_total)

# Gains (Cold vs Total)
gain_xgb_cold = area_tl_cold / area_xgb_cold
gain_xgb_total = area_tl_total / area_xgb_total

# 3. VISUALIZATION
plt.figure(figsize=(12, 7))

# Plot lines
plt.plot(labels, res_local_xgb, label='Local XGB (2018 Only)', color='#e74c3c', marker='o', linestyle='--')
plt.plot(labels, res_mlp, label='Local MLP (2018 Only)', color='#95a5a6', marker='d', linestyle='--')
plt.plot(labels, res_tl, label='Adaptive TL (Path Transfer)', color='#2980b9', marker='^', linewidth=3)

# Shading Phase 1: Cold Start (The Core Defense Contribution)
plt.fill_between(labels[:4], 0, res_tl[:4], color='#2980b9', alpha=0.1, label='Cold Start Utility Area')
plt.fill_between(labels[:4], res_local_xgb[:4], res_tl[:4], where=(np.array(res_tl[:4]) >= np.array(res_local_xgb[:4])), 
                 color='green', alpha=0.2, hatch='//', label=f'Cold Start Gain: {gain_xgb_cold:.2f}x')

# Shading Phase 2: Maturation
plt.fill_between(labels[3:], res_local_xgb[3:], res_tl[3:], color='gray', alpha=0.05, label='Convergence Zone')

# Phase Boundary
plt.axvline(x=50, color='#c0392b', linestyle='-.', alpha=0.6)
plt.text(12, 0.75, "PHASE 1: COLD START\n(Transfer Advantage)", fontsize=10, fontweight='bold', color='#2c3e50')
plt.text(60, 0.75, "PHASE 2: MATURATION\n(Model Convergence)", fontsize=10, fontweight='bold', color='#7f8c8d')

# Professional Formatting
plt.title("Longitudinal Security Advantage: Dual-Phase Transfer Gain", fontsize=15, fontweight='bold')
plt.xlabel("Number of Labeled 2018 Samples ($n$)", fontsize=12)
plt.ylabel("Detection Efficacy (F1-Score)", fontsize=12)
plt.legend(loc='lower right', frameon=True, shadow=True, fontsize=9)
plt.grid(alpha=0.3, linestyle='--')
plt.ylim(0, 0.85)
plt.tight_layout()

plt.savefig('dual_phase_transfer_analysis.png', dpi=300)
plt.show()

print(f"--- COLD START GAIN (n=0 to 50) ---")
print(f"TL Advantage vs XGB: {gain_xgb_cold:.2f}x")
print(f"\n--- TOTAL LIFE-CYCLE GAIN (n=0 to 100) ---")
print(f"TL Advantage vs XGB: {gain_xgb_total:.2f}x")