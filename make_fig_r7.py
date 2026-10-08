"""Figure for Round 7: (a) AQT attack-F1 vs TPR at 1% FPR (noise-degraded and natural points), (b) label-budget curve,
(c) decoy recall vs decoy rate for three attackers, (d) burst timeline of the replay (Wed 21 Feb)."""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
C = {"blue": "#1F5FBF", "orange": "#C8561B", "green": "#2E7D5B", "gray": "#6B7280", "purple": "#7A4FB5"}
fig, ax = plt.subplots(1, 4, figsize=(7.1, 2.15))

# (a) dose-response
d = pd.read_csv("round7_dose_summary.csv")
mk = {"tue20_18": ("o", C["orange"], "Tue 20"), "wed14_18": ("s", C["purple"], "Wed 14"), "wed21_18": ("^", C["blue"], "Wed 21 (multi)"), "wed21_raw": ("D", C["green"], "Wed 21 (Wed pair)")}
for day, g in d.groupby("day"):
    m, c, lab = mk[day]
    ax[0].plot(g.rprec, g.aqt_af1, "-", color=c, lw=0.8, alpha=0.5)
    ax[0].plot(g.rprec, g.aqt_af1, m, color=c, ms=3, label=lab)
    g0 = g[g.sigma == 0]; ax[0].plot(g0.rprec, g0.aqt_af1, m, color=c, ms=6.5, mfc="none", mew=1.0)
ax[0].set_xlabel("R-precision (frozen)"); ax[0].set_ylabel("AQT attack-F1"); ax[0].set_title("(a) AQT vs R-precision")
ax[0].legend(frameon=False, loc="upper left", handletextpad=0.2, borderpad=0.1)

# (b) label budget
L = pd.read_csv("round7_labels_multi77_macro_f1_summary.csv"); L = L[L.kind == "level"]
fr = [0.001, 0.01, 0.05, 0.1, 0.25, 1.0]
for m, c in ((1, C["blue"]), (10, C["orange"])):
    y = [L[L.a == f"lab{f}_adam_{m}x"]["mean"].iloc[0] for f in fr]; lo = [L[L.a == f"lab{f}_adam_{m}x"]["lo"].iloc[0] for f in fr]; hi = [L[L.a == f"lab{f}_adam_{m}x"]["hi"].iloc[0] for f in fr]
    ax[1].plot(np.array(fr) * 100, y, "o-", color=c, ms=2.5, lw=1, label=f"{m}x rate")
    ax[1].fill_between(np.array(fr) * 100, lo, hi, color=c, alpha=0.15, lw=0)
    ya = L[L.a == f"alert_adam_{m}x"]["mean"].iloc[0]; ax[1].plot([100], [ya], "x", color=c, ms=5, mew=1.2)
ax[1].axhline(L[L.a == "aqt"]["mean"].iloc[0], color=C["gray"], ls="--", lw=0.8); ax[1].text(0.12, L[L.a == "aqt"]["mean"].iloc[0] + 0.008, "AQT", color=C["gray"], fontsize=7)
ax[1].set_xscale("log"); ax[1].set_xlabel("% of flows labelled"); ax[1].set_ylabel("macro-F1"); ax[1].set_title("(b) Label budget")
ax[1].legend(frameon=False, loc="upper left", handletextpad=0.3, borderpad=0.1)

# (c) decoys
D = pd.read_csv("round6_decoy_raw77_summary.csv"); D = D[D.rule == "aqt"]
for att, c in (("random", C["gray"]), ("surrogate", C["orange"]), ("victim", C["blue"])):
    r = D[D.attacker == att].iloc[0]; xs = [0, 0.5, 1, 2, 5, 10]; ys = [r["0.0"], r["0.005"], r["0.01"], r["0.02"], r["0.05"], r["0.1"]]
    ax[2].plot(xs, ys, "o-", ms=2.5, lw=1, color=c, label={"random": "random", "surrogate": "surrogate", "victim": "victim-aware"}[att])
S = json.load(open("round8_surrogate.json"))
xs3 = [0, 1, 5]; ys3 = [np.mean([r["recall"] for r in S[f"disjoint_B|{x/100}"]]) for x in xs3]
ax[2].plot(xs3, ys3, "s--", ms=3, lw=1, color=C["green"], label="disjoint-data surrogate")
ax[2].set_xlabel("decoy rate (% of traffic)"); ax[2].set_ylabel("recall on real attacks"); ax[2].set_title("(c) Decoy inflation"); ax[2].set_ylim(0, 0.65)
ax[2].legend(frameon=False, loc="lower left", handletextpad=0.3, borderpad=0.1)

# (d) burst timeline
T = json.load(open("round7_cleanwin_multi77_raw.json"))["timeline"]["wed21_18"]
t = np.array(T["t"]) / 1000.0
a = ax[3]
a.plot(t, np.array(T["local_prev"]) * 100, color=C["orange"], lw=1, label="attack share")
a.plot(t, np.array(T["alert_rate_500"]) * 100, color=C["blue"], lw=1, label="AQT alert rate")
a.axhline(1, color=C["gray"], ls="--", lw=0.7)
a.set_xlabel("flows (thousands)"); a.set_ylabel("% of last 500 flows"); a.set_title("(d) Wed 21 burst")
a.legend(frameon=False, loc="upper right", handletextpad=0.3, borderpad=0.1)
plt.tight_layout(pad=0.4, w_pad=0.6); plt.savefig("/home/claude/manuscript/fig_r7.pdf"); plt.savefig("/home/claude/manuscript/fig_r7.png", dpi=170)
print("saved")
