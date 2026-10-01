import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("verified_ablation_raw.json") as f:
    main = json.load(f)
with open("oracle_robustness_raw.json") as f:
    robust = json.load(f)
combined = {**main, **robust}

checkpoints = [1000, 5000, 10000, 20000]  # n=100 (pre-Phase-2, huge variance) omitted; all configs are statistically indistinguishable there

# Okabe-Ito colorblind-safe palette
# Each entry: (config_id, label, color, linestyle, marker, markersize, zorder)
SERIES = [
    ("1_base_tl", "Base TL (no adapt)", "#666666", "-", "o", 4, 2),
    ("3_tl_aqt", "TL + AQT (no fine-tune)", "#0072B2", "-", "o", 4, 2),
    ("5a_full_atl_oracle", "Oracle, original (LR=1x, unweighted)", "#D55E00", "-", "o", 4, 2),
    ("5a_oracle_lowlr", "Oracle, LR x0.1", "#D55E00", "--", "o", 4, 3),
    # Drawn last, on top, with large hollow markers and a sparse dotted line so it
    # is visible where it sits exactly on top of the original oracle (see caption).
    ("5a_oracle_classweighted", "Oracle, 99:1 class-weighted (identical to original, see note)", "#CC79A7", (0, (1, 4)), "s", 8, 5),
    ("5b_full_atl_pseudo", "Pseudo-label", "#009E73", "-", "o", 4, 2),
]

fig, ax = plt.subplots(figsize=(9, 6))
for cfg_id, label, color, style, marker, msize, z in SERIES:
    means = np.array([np.mean(combined[cfg_id][str(n)]) for n in checkpoints])
    stds = np.array([np.std(combined[cfg_id][str(n)]) for n in checkpoints])
    mfc = "none" if cfg_id == "5a_oracle_classweighted" else color
    mew = 2 if cfg_id == "5a_oracle_classweighted" else 1
    ax.plot(checkpoints, means, color=color, linestyle=style, linewidth=2.2 if cfg_id != "5a_oracle_classweighted" else 1.6,
            marker=marker, markersize=msize, markerfacecolor=mfc, markeredgewidth=mew, markeredgecolor=color,
            label=label, zorder=z)
    ax.fill_between(checkpoints, means - stds, means + stds, color=color, alpha=0.08, zorder=1)

ax.annotate(
    "Orange (solid) & magenta overlap exactly:\nclass-weighting is bit-identical to\nthe original oracle run, 30/30 seeds",
    xy=(5000, np.mean(combined["5a_full_atl_oracle"]["5000"])),
    xytext=(6200, 0.5245),
    fontsize=8.5, color="#8a3d00", ha="left",
    arrowprops=dict(arrowstyle="->", color="#8a3d00", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#8a3d00", lw=0.8, alpha=0.95),
)

ax.set_xscale("log")
ax.set_xlim(900, 22000)
ax.set_ylim(0.48, 0.535)
ax.set_xlabel("Cold-start stream position (n samples)")
ax.set_ylabel("Macro-F1 (mean ± std, 30 Monte Carlo runs)")
ax.set_title("Oracle fine-tuning: sensitivity to learning rate and class weighting", fontsize=12, fontweight="bold")
ax.legend(fontsize=8.5, loc="lower left", framealpha=0.95)
ax.grid(alpha=0.25)
ax.set_xticks(checkpoints)
ax.set_xticklabels([str(c) for c in checkpoints])

plt.tight_layout()
plt.savefig("oracle_robustness.png", dpi=150)
print("saved oracle_robustness.png")
