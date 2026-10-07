"""Summarize round7_replay_ft outputs: mean attack-F1/recall/precision per day x order x variant over seeds."""
import json, glob, collections
import numpy as np, pandas as pd
rows = []
for f in sorted(glob.glob("round7_replayft_*.json")):
    pool, seed = f[:-5].split("_")[-2:]
    for k, v in json.load(open(f)).items():
        day, order, var = k.split("|")
        rows.append(dict(pool=pool, seed=int(seed), day=day, order=order, variant=var, **v))
df = pd.DataFrame(rows)
g = df.groupby(["pool", "day", "order", "variant"])[["recall", "precision", "attack_f1", "alert_rate"]].mean()
n = df.groupby(["pool", "day", "order", "variant"]).seed.nunique().rename("n_seeds")
out = g.join(n).round(3).reset_index()
out.to_csv("round7_replayft_summary.csv", index=False)
print(out.to_string())
