"""Held-out rate tuning: choose the persistent-Adam rate multiplier on tuning seeds 3000-3005 (disjoint from the 20
evaluation seeds), report the chosen rate's gain over AQT on the 20 evaluation seeds.
Two families: true labels (oracle_adam, tune files round7_tune_*, eval round7_cells_* and earlier shards) and
pseudo-labels (pseudo_adam, tune files round7_r8tune_*, eval round7_r8_*) -> round7_tuned_rate_summary.csv"""
import glob, json, os, numpy as np, pandas as pd
from scipy import stats
OLD = {"paper10": "multiseed_paper10", "multi77": "multiday_multi77", "multi10": "multiday_multi10", "raw77": None, "nf": None}
NEW = {"paper10": "round7_cells_paper10", "multi77": "round7_cells_multi77", "multi10": "round7_cells_multi10", "raw77": "round7_cells_raw77", "nf": "round7_cells_nf"}
RATES = [1, 3, 10, 30, 100]
def load(prefix):
    d = {}
    for f in sorted(glob.glob(f"{prefix}_shard*.json")): d.update(json.load(open(f)))
    return d
rows = []
for fam, tune_pre, eval_pre in (("oracle", "round7_tune", None), ("pseudo", "round7_r8tune", "round7_r8")):
    for pool in ("paper10", "multi77", "multi10", "raw77", "nf"):
        tune = load(f"{tune_pre}_{pool}")
        if not tune: continue
        var = f"{fam}_adam"
        gain = {m: np.array([rec[f"{var}_{m}x"]["20000"]["macro_f1"] - rec["aqt"]["20000"]["macro_f1"] for rec in tune.values()]) for m in RATES}
        best = max(RATES, key=lambda m: gain[m].mean())
        ev = {}
        srcs = (OLD[pool], NEW[pool]) if fam == "oracle" else (f"{eval_pre}_{pool}", NEW[pool])
        for src in srcs:
            if src:
                for key, rec in load(src).items():
                    ev.setdefault(key, {}).update({v: m["20000"]["macro_f1"] for v, m in rec.items()})
        per = {}
        for key, r in ev.items():
            if f"{var}_{best}x" in r and "aqt" in r:
                per.setdefault(int(key.split("_")[0]), []).append(r[f"{var}_{best}x"] - r["aqt"])
        if len(per) < 3: continue
        x = np.array([np.mean(v) for _, v in sorted(per.items())]); m = x.mean(); h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        rows.append(dict(family=fam, pool=pool, n_tune=len(tune), tuning_gains={k: round(float(v.mean()), 4) for k, v in gain.items()}, chosen=best,
                         heldout_gain=m, lo=m - h, hi=m + h, seeds_positive=int((x > 0).sum()), n_eval=len(x)))
out = pd.DataFrame(rows); out.to_csv("round7_tuned_rate_summary.csv", index=False)
pd.set_option("display.width", 240, "display.max_colwidth", 120)
print(out.round(4).to_string())
