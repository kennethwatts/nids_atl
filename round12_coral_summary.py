"""Summarize round12_coral_<pool>.json: per pool mean over seeds of AUROC, R-precision and AQT attack-F1 for none / coral2000 /
coralpool, plus paired differences (CORAL minus none) with 95% t-intervals across the 5 seeds (streams averaged within seed).
usage: python3 round12_coral_summary.py -> round12_coral_summary.csv"""
import json, numpy as np, pandas as pd
from scipy import stats
rows = []
for pool in ("paper10", "raw77", "multi77", "multi10", "nf"):
    try: d = json.load(open(f"round12_coral_{pool}.json"))
    except FileNotFoundError: continue
    seeds = sorted({x["seed"] for x in d})
    for met in ("auroc", "rprec", "af1", "alert"):
        per = {v: np.array([np.mean([x[v][met] for x in d if x["seed"] == s]) for s in seeds]) for v in ("none", "coral2000", "coralpool")}
        for v in ("coral2000", "coralpool"):
            df = per[v] - per["none"]; n = len(df); se = df.std(ddof=1) / np.sqrt(n); t = stats.t.ppf(0.975, n - 1)
            rows.append(dict(pool=pool, metric=met, none=per["none"].mean(), variant=v, mean=per[v].mean(), diff=df.mean(),
                             lo=df.mean() - t * se, hi=df.mean() + t * se, n_seeds=n))
out = pd.DataFrame(rows).round(3); out.to_csv("round12_coral_summary.csv", index=False)
print(out.to_string(index=False))
