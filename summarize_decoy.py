"""Summarize adversarial Part A (decoy sweep): mean recall on real attacks,
precision, alert rate per rule and decoy rate, with seed-cluster bootstrap CI on recall."""
import json, sys
import numpy as np, pandas as pd
from summarize_grid import cluster_boot

rows = []
for pool in ["raw77", "paper10"]:
    d = json.load(open(f"adversarial_decoy_{pool}_raw.json"))
    for key, rules in d.items():
        seed, j, rate = key.split("_")
        for r, m in rules.items():
            rows.append(dict(pool=pool, seed=int(seed), stream=int(j), rate=float(rate), rule=r, **m))
df = pd.DataFrame(rows)
out = []
for (pool, rate, rule), g in df.groupby(["pool", "rate", "rule"]):
    w = g[["seed", "recall"]].copy()
    lo, hi = cluster_boot(w, lambda x: x["recall"].mean(), reps=400)
    out.append(dict(pool=pool, decoy_rate=rate, rule=rule, recall=g.recall.mean(), ci_lo=lo, ci_hi=hi,
                    precision=g.precision.mean(), alert_rate=g.alert_rate.mean()))
o = pd.DataFrame(out)
o.to_csv("adversarial_decoy_summary.csv", index=False)
pd.set_option("display.width", 200)
for pool in ["raw77", "paper10"]:
    print(pool)
    print(o[o.pool == pool].pivot(index="decoy_rate", columns="rule", values="recall").round(3))
    print(o[o.pool == pool].pivot(index="decoy_rate", columns="rule", values="alert_rate").round(4))
