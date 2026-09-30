import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("oracle_diagnostic.json") as f:
    d = json.load(f)

# Okabe-Ito colorblind-safe palette
COLORS = {
    "1_base_tl": "#666666",          # neutral gray (reference / no adaptation)
    "3_tl_aqt": "#0072B2",           # blue
    "5a_full_atl_oracle": "#D55E00", # vermillion
    "5b_full_atl_pseudo": "#009E73", # green
}
LABELS = {
    "1_base_tl": "Base TL (no adapt)",
    "3_tl_aqt": "TL + AQT (no adapt)",
    "5a_full_atl_oracle": "Full ATL, oracle labels",
    "5b_full_atl_pseudo": "Full ATL, pseudo-label (proposed)",
}

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Why oracle fine-tuning degrades: per-step trajectories, mean of 10 Monte Carlo streams (n=20,000)", fontsize=12, fontweight="bold")

# Panel A: raw predicted probability, attack vs benign, log scale
ax = axes[0, 0]
for cfg in ["3_tl_aqt", "5a_full_atl_oracle", "5b_full_atl_pseudo"]:
    steps = d[cfg]["step"]
    ax.plot(steps, d[cfg]["mean_p_attack_mean"], color=COLORS[cfg], linewidth=2, label=f"{LABELS[cfg]} - attack")
    ax.plot(steps, d[cfg]["mean_p_benign_mean"], color=COLORS[cfg], linewidth=1, linestyle="--", label=f"{LABELS[cfg]} - benign")
ax.set_yscale("log")
ax.set_xlabel("Stream step")
ax.set_ylabel("Mean raw model output (log scale)")
ax.set_title("A. Raw probability output collapses under oracle updates")
ax.legend(fontsize=7, loc="lower left")
ax.grid(alpha=0.25)

# Panel B: output-layer bias (decision function drift)
ax = axes[0, 1]
for cfg in ["1_base_tl", "3_tl_aqt", "5a_full_atl_oracle", "5b_full_atl_pseudo"]:
    steps = d[cfg]["step"]
    ax.plot(steps, d[cfg]["output_bias_mean"], color=COLORS[cfg], linewidth=2, label=LABELS[cfg])
ax.set_xlabel("Stream step")
ax.set_ylabel("Output-layer bias term")
ax.set_title("B. Decision function is pushed toward \"always benign\"")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

# Panel C: AQT threshold trajectory
ax = axes[1, 0]
for cfg in ["3_tl_aqt", "5a_full_atl_oracle", "5b_full_atl_pseudo"]:
    steps = d[cfg]["step"]
    ax.plot(steps, d[cfg]["tau_mean"], color=COLORS[cfg], linewidth=2, label=LABELS[cfg])
ax.set_yscale("log")
ax.set_xlabel("Stream step")
ax.set_ylabel("AQT threshold tau (log scale)")
ax.set_title("C. AQT threshold chases the collapse down")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

# Panel D: fraction predicted positive (rolling), vs true 1% rate
ax = axes[1, 1]
for cfg in ["1_base_tl", "3_tl_aqt", "5a_full_atl_oracle", "5b_full_atl_pseudo"]:
    steps = d[cfg]["step"]
    ax.plot(steps, d[cfg]["frac_pred_positive_mean"], color=COLORS[cfg], linewidth=2, label=LABELS[cfg])
ax.axhline(0.01, color="black", linewidth=1, linestyle=":", label="true malicious rate (1%)")
ax.set_xlabel("Stream step")
ax.set_ylabel("Fraction of recent 500 predictions positive")
ax.set_title("D. Predicted-positive rate stays near 1% regardless")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("oracle_diagnostic.png", dpi=150)
print("saved oracle_diagnostic.png")
