import matplotlib.pyplot as plt
import numpy as np

# Data points from your successful runs
labels = [0, 5, 10, 15, 20, 30, 50, 100]
res_local_rf = [0.000, 0.010, 0.261, 0.280, 0.519, 0.565, 0.628, 0.782]
res_tl = [0.374, 0.409, 0.346, 0.466, 0.594, 0.599, 0.596, 0.697]
res_legacy_adaptive = [0.374] * len(labels)

plt.figure(figsize=(10, 6))

# Plot lines
plt.plot(labels, res_legacy_adaptive, label='Legacy (2017 + Adaptive Thresh)', color='#f39c12', linestyle=':', linewidth=2.5)
plt.plot(labels, res_local_rf, label='Local RF (2018 Only)', color='#e74c3c', marker='o', linestyle='--')
plt.plot(labels, res_tl, label='Proposed Adaptive TL', color='#2980b9', marker='^', linewidth=3.5)

# Fill the "Security Gap"
plt.fill_between(labels, res_local_rf, res_tl, where=(np.array(res_tl) > np.array(res_local_rf)), 
                 color='green', alpha=0.15, label='TL Advantage (Coverage Gap)')

# Add Zero-Shot Annotation
plt.annotate('Zero-Shot Protection (0.374)', xy=(0, 0.374), xytext=(15, 0.15),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
             fontsize=10, fontweight='bold')

plt.title("Conclusion: Bridging the 'Deployment Gap' with Transfer Learning", fontsize=14, fontweight='bold')
plt.xlabel("Number of Manual Labels (New Deployment)", fontsize=12)
plt.ylabel("Detection Efficacy (F1-Score)", fontsize=12)
plt.ylim(-0.05, 1.0)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('thesis_conclusion_final.png', dpi=300)
plt.show()