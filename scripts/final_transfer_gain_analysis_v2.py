import numpy as np
import matplotlib.pyplot as plt

# 1. DATA FROM YOUR SUCCESSFUL RUN
labels = [0, 10, 20, 50, 100]
res_local = [0.000, 0.375, 0.444, 0.596, 0.710]
res_tl = [0.390, 0.522, 0.522, 0.583, 0.686]
res_legacy = [0.390] * len(labels)

# 2. CALCULATE INTEGRATED AREA (Transfer Gain)
x_gap = np.array(labels[:4]) 
y_tl = np.array(res_tl[:4])
y_local = np.array(res_local[:4])

# Fixed DeprecationWarning for Python 3.13
area_tl = np.trapezoid(y_tl, x_gap)
area_local = np.trapezoid(y_local, x_gap)
transfer_gain = area_tl - area_local

# 3. VISUALIZATION
plt.figure(figsize=(10, 6))
plt.plot(labels, res_legacy, label='Legacy (2017 Expert)', color='#f39c12', linestyle=':', linewidth=2)
plt.plot(labels, res_local, label='Local XGB (2018 Only)', color='#e74c3c', marker='o', linestyle='--')
plt.plot(labels, res_tl, label='Adaptive TL (Path Transfer)', color='#2980b9', marker='^', linewidth=3)

# Shading with interpolate=True fills the gap exactly up to the crossover point
plt.fill_between(labels, res_local, res_tl, where=(np.array(res_tl) >= np.array(res_local)), 
                 color='green', alpha=0.2, interpolate=True, label=f'Transfer Gain (Area: {transfer_gain:.2f})')

plt.title("Security Advantage: Transfer Learning vs. Local Training", fontsize=14, fontweight='bold')
plt.xlabel("Number of Manual Labels (Deployment Age)", fontsize=12)
plt.ylabel("Detection Efficacy (F1-Score)", fontsize=12)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('final_transfer_gain_analysis_v2.png', dpi=300)
print(f"Total Transfer Gain Area (0-50): {transfer_gain:.4f}")
plt.show()