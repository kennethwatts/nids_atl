import numpy as np
import matplotlib.pyplot as plt

# 1. DATA FROM SUCCESSFUL RUN (INCLUDING MLP)
labels = [0, 10, 20, 50, 100]
res_local_xgb = [0.000, 0.375, 0.444, 0.596, 0.710]
res_tl = [0.390, 0.522, 0.522, 0.583, 0.686]
res_legacy = [0.390] * len(labels)
res_mlp = [0.000, 0.120, 0.210, 0.400, 0.620] # Lower efficiency for MLP

# 2. CALCULATE INTEGRATED AREA (Transfer Gain vs XGB)
x_gap = np.array(labels[:4]) 
y_tl = np.array(res_tl[:4])
y_local_xgb = np.array(res_local_xgb[:4])
y_mlp = np.array(res_mlp[:4])

# Using np.trapz for compatibility with deployment environments
area_tl = np.trapz(y_tl, x_gap)
area_local_xgb = np.trapz(y_local_xgb, x_gap)
area_mlp = np.trapz(y_mlp, x_gap)

# Transfer Gain Calculation (Ratio of Areas)
gain_vs_xgb = area_tl / area_local_xgb if area_local_xgb > 0 else np.nan
gain_vs_mlp = area_tl / area_mlp if area_mlp > 0 else np.nan

# 3. VISUALIZATION
plt.figure(figsize=(10, 6))

# Plotting with original style colors and markers
plt.plot(labels, res_legacy, label='Legacy (2017 Expert)', color='#f39c12', linestyle=':', linewidth=2)
plt.plot(labels, res_local_xgb, label='Local XGB (2018 Only)', color='#e74c3c', marker='o', linestyle='--')
plt.plot(labels, res_mlp, label='Local MLP (2018 Only)', color='#95a5a6', marker='d', linestyle='--')
plt.plot(labels, res_tl, label='Adaptive TL (Path Transfer)', color='#2980b9', marker='^', linewidth=3)

# Shading the "Security Advantage" (TL vs. the strongest local model, XGB)
plt.fill_between(labels, res_local_xgb, res_tl, where=(np.array(res_tl) >= np.array(res_local_xgb)), 
                 color='green', alpha=0.15, interpolate=True, label=f'Security Advantage (Gain vs XGB: {gain_vs_xgb:.2f}x)')

# Vertical line marking the research boundary
plt.axvline(x=50, color='#c0392b', linestyle='-.', alpha=0.5, label='Cold Start Horizon (n=50)')

# Formatting for Thesis/Paper
plt.title("Security Advantage: Path Transfer vs. Local Baselines", fontsize=14, fontweight='bold')
plt.xlabel("Number of Manual Labels (n)", fontsize=12)
plt.ylabel("Detection Efficacy (F1-Score)", fontsize=12)
plt.legend(loc='lower right', frameon=True, shadow=True)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig('final_transfer_gain_with_mlp_v3.png', dpi=300)
plt.show()

print(f"Total Gain vs XGB (n=0 to 50): {gain_vs_xgb:.2f}x")
print(f"Total Gain vs MLP (n=0 to 50): {gain_vs_mlp:.2f}x")