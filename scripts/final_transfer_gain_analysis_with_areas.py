import numpy as np
import matplotlib.pyplot as plt

# 1. FINAL AVERAGED EXPERIMENTAL DATA
labels = [0, 10, 20, 50, 100]
res_local_xgb = [0.000, 0.023, 0.271, 0.588, 0.761]
res_mlp = [0.000, 0.298, 0.439, 0.641, 0.766]
res_tl = [0.433, 0.447, 0.505, 0.640, 0.759]
res_legacy = [res_tl[0]] * len(labels)

# 2. CALCULATE AREAS (Integral from n=0 to n=50)
x_gap = np.array(labels[:4]) 
y_tl, y_xgb, y_mlp = np.array(res_tl[:4]), np.array(res_local_xgb[:4]), np.array(res_mlp[:4])

area_tl = np.trapz(y_tl, x_gap)
area_xgb = np.trapz(y_xgb, x_gap)
area_mlp = np.trapz(y_mlp, x_gap)

# Calculate Gains (Ratios)
gain_xgb = area_tl / area_xgb if area_xgb > 0 else float('inf')
gain_mlp = area_tl / area_mlp if area_mlp > 0 else float('inf')

# 3. VISUALIZATION
plt.figure(figsize=(11, 7))

# Plot performance lines with Area in labels
plt.plot(labels, res_legacy, label=f'Legacy (Area: {res_tl[0]*50:.1f})', color='#f39c12', linestyle=':', linewidth=2)
plt.plot(labels, res_local_xgb, label=f'Local XGB (Area: {area_xgb:.1f})', color='#e74c3c', marker='o', linestyle='--')
plt.plot(labels, res_mlp, label=f'Local MLP (Area: {area_mlp:.1f})', color='#95a5a6', marker='d', linestyle='--')
plt.plot(labels, res_tl, label=f'Adaptive TL (Area: {area_tl:.1f})', color='#2980b9', marker='^', linewidth=3)

# Shading & Gain Ratios
plt.fill_between(labels, res_local_xgb, res_tl, where=(np.array(res_tl) >= np.array(res_local_xgb)), 
                 color='green', alpha=0.1, interpolate=True, label=f'Gain vs XGB: {gain_xgb:.2f}x')

plt.fill_between(labels, res_mlp, res_tl, where=(np.array(res_tl) >= np.array(res_mlp)), 
                 color='#27ae60', alpha=0.2, interpolate=True, label=f'Gain vs MLP: {gain_mlp:.2f}x')

plt.axvline(x=50, color='#c0392b', linestyle='-.', alpha=0.5, label='Cold Start Horizon (n=50)')

plt.title("Security Advantage: Path Transfer vs. Local Baselines", fontsize=15, fontweight='bold')
plt.xlabel("Number of Labeled 2018 Samples (Deployment Age)", fontsize=12)
plt.ylabel("Detection Efficacy (F1-Score)", fontsize=12)
plt.legend(loc='lower right', frameon=True, shadow=True, fontsize=10)
plt.grid(alpha=0.3, linestyle='--')
plt.ylim(-0.05, 0.85)
plt.tight_layout()

plt.savefig('final_transfer_gain_analysis_with_areas.png', dpi=300)
plt.show()
