"""Held-out rate tuning: choose the persistent-Adam rate multiplier on tuning seeds 3000-3005 (disjoint from the 20
evaluation seeds), report the chosen rate's gain over AQT on the 20 evaluation seeds."""
import glob, json, numpy as np, pandas as pd
from scipy import stats
OLD = {"paper10": "multiseed_paper10", "multi77": "multiday_multi77", "multi10": "multiday_multi10", "raw77": None}
NEW = {"paper10": "round7_cells_paper10", "multi77": "round7_cells_multi77", "multi10": "round7_cells_multi10", "raw77": "round7_cells_raw77"}
RATES = [1, 3, 10, 30, 100]
def load(prefix):
    d = {}
    for f in sorted(glob.glob(f"{prefix}_shard*.json")): d.update(json.load(open(f)))
    return d
rows = []
for pool in ("paper10", "multi77", "multi10", "raw77"):
    tune = load(f"round7_tune_{pool}")
    if not tune: continue
    gain = {m: np.array([rec[f"oracle_adam_{m}x"]["20000"]["macro_f1"] - rec["aqt"]["20000"]["macro_f1"] for rec in tune.values()]) for m in RATES}
    best = max(RATES, key=lambda m: gain[m].mean())
    ev = {}
    for src in (OLD[pool], NEW[pool]):
        if src:
            for key, rec in load(src).items():
                ev.setdefault(key, {}).update({v: m["20000"]["macro_f1"] for v, m in rec.items()})
    per = {}
    for key, r in ev.items():
        if f"oracle_adam_{best}x" in r and "aqt" in r:
            per.setdefault(int(key.split("_")[0]), []).append(r[f"oracle_adam_{best}x"] - r["aqt"])
    x = np.array([np.mean(v) for _, v in sorted(per.items())]); m = x.mean(); h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    rows.append(dict(pool=pool, n_tune=len(tune), tuning_gains={k: round(float(v.mean()), 4) for k, v in gain.items()}, chosen=best,
                     heldout_gain=m, lo=m - h, hi=m + h, seeds_positive=int((x > 0).sum()), n_eval=len(x)))
out = pd.DataFrame(rows); out.to_csv("round7_tuned_rate_summary.csv", index=False)
pd.set_option("display.width", 220, "display.max_colwidth", 120)
print(out.round(4).to_string())
