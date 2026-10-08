"""Mean attack-F1/recall/precision over seeds for round8_replaythr_*.json (day x order x variant x rule)."""
import json, glob, pandas as pd
rows = []
for f in sorted(glob.glob("round8_replaythr_*.json")):
    pool, seed = f[:-5].split("_")[-2:]
    for k, v in json.load(open(f)).items():
        day, order, var, rule = k.split("|"); rows.append(dict(pool=pool, seed=int(seed), day=day, order=order, variant=var, rule=rule, **v))
df = pd.DataFrame(rows)
g = df.groupby(["pool", "day", "order", "variant", "rule"])[["recall", "precision", "attack_f1"]].mean().round(3).reset_index()
g["n_seeds"] = df.groupby(["pool", "day", "order", "variant", "rule"]).seed.nunique().values
g.to_csv("round8_replaythr_summary.csv", index=False)
x = g[(g.order == "time_ordered") & g.variant.isin(["frozen", "oracle_adam_1x", "oracle_adam_10x"])]
print(x.pivot_table(index=["pool", "day", "variant"], columns="rule", values="attack_f1").to_string())
print(x.pivot_table(index=["pool", "day", "variant"], columns="rule", values="recall").to_string())
