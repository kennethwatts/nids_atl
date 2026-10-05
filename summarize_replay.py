"""Summarize time-ordered replay vs shuffled control (means over 3 seeds x 10 thinning draws)."""
import json
import numpy as np, pandas as pd
d = json.load(open("time_ordered_replay_raw.json")); rows = []
for order, seeds in d.items():
    for seed, recs in seeds.items():
        for i, rec in enumerate(recs):
            for r, m in rec.items():
                if r == "_cluster":
                    rows.append(dict(order=order, seed=seed, rule="_cluster", **m))
                else:
                    rows.append(dict(order=order, seed=seed, rule=r, **m))
df = pd.DataFrame(rows)
res = df[df.rule != "_cluster"].groupby(["order", "rule"])[["recall", "precision", "attack_f1", "alert_rate"]].mean().round(3)
cl = df[df.rule == "_cluster"].groupby("order")[["share_attacks_in_densest_5pct", "peak_local_prevalence_500"]].mean().round(3)
seedrec = df[(df.rule == "aqt")].groupby(["order", "seed"])["recall"].mean().round(3)
pd.set_option("display.width", 200)
print(res); print(cl); print(seedrec)
res.to_csv("time_ordered_replay_summary.csv"); cl.to_csv("time_ordered_replay_cluster.csv")
