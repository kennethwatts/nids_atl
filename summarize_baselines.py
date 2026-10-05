"""Summarize threshold_baselines: mean metrics per pool/rule/prevalence with seed-cluster bootstrap CI on attack-F1."""
import json
import pandas as pd
from summarize_grid import cluster_boot

rows = []
for pool, f in [("raw77_logit", "threshold_baselines_raw77_logit_raw.json"), ("paper10_prob", "threshold_baselines_paper10_prob_raw.json")]:
    for key, rules in json.load(open(f)).items():
        seed, j, prev = key.split("_")
        for r, m in rules.items():
            rows.append(dict(pool=pool, seed=int(seed), prev=float(prev), rule=r, **m))
df = pd.DataFrame(rows); out = []
for (pool, prev, rule), g in df.groupby(["pool", "prev", "rule"]):
    lo, hi = cluster_boot(g[["seed", "attack_f1"]], lambda x: x["attack_f1"].mean(), reps=300)
    out.append(dict(pool=pool, prevalence=prev, rule=rule, attack_f1=g.attack_f1.mean(), f1_lo=lo, f1_hi=hi,
                    recall=g.recall.mean(), precision=g.precision.mean(), alert_rate=g.alert_rate.mean(), macro_f1=g.macro_f1.mean()))
o = pd.DataFrame(out); o.to_csv("threshold_baselines_summary.csv", index=False)
pd.set_option("display.width", 220)
for pool in o.pool.unique():
    for m in ["attack_f1", "recall", "alert_rate"]:
        print(pool, m); print(o[o.pool == pool].pivot(index="prevalence", columns="rule", values=m).round(3))
