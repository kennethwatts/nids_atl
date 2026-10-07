import json, numpy as np, pandas as pd
rows = []
for pool in ("multi77", "raw77"):
    d = json.load(open(f"round7_cleanwin_{pool}_raw.json"))
    for day, seeds in d["replay"].items():
        recs = [r for s in seeds.values() for r in s]
        for rule in recs[0]:
            rows.append(dict(pool=pool, day=day, rule=rule, **{m: float(np.mean([r[rule][m] for r in recs])) for m in ("recall", "precision", "attack_f1", "alert_rate")}))
df = pd.DataFrame(rows); df.to_csv("round7_cleanwin_summary.csv", index=False)
rk = []
for pool in ("multi77", "raw77"):
    d = json.load(open(f"round7_cleanwin_{pool}_raw.json"))["rank"]
    for day, seeds in d.items():
        for fam in list(seeds.values())[0]:
            rk.append(dict(pool=pool, day=day, family=fam, **{m: float(np.mean([s[fam][m] for s in seeds.values()])) for m in ("auroc", "tpr1")}))
pd.DataFrame(rk).to_csv("round7_ranking_summary.csv", index=False)
pd.set_option("display.width", 200)
piv = df[df.rule.str.startswith("win_k") & df.rule.str.endswith("_c0.0")].copy()
piv["k"] = piv.rule.str.extract(r"k(\d+)").astype(int)
for m in ("recall", "precision", "alert_rate"):
    print(m); print(piv.pivot_table(index="k", columns=["pool", "day"], values=m).round(3).to_string())
ref = df[df.rule.isin(["static_ceiling", "src_q99", "budget_1pct", "aqt_W500"])]
print(ref.pivot_table(index="rule", columns=["pool", "day"], values="recall").round(3).to_string())
print(ref.pivot_table(index="rule", columns=["pool", "day"], values="precision").round(3).to_string())
c = df[df.rule.str.startswith("win_k") & df.rule.str.contains("_c0.0[15]")].copy()
c["k"] = c.rule.str.extract(r"k(\d+)").astype(int); c["c"] = c.rule.str.extract(r"_c([\d.]+)")
print("contamination recall"); print(c[c.k.isin([500, 2000])].pivot_table(index=["k", "c"], columns=["pool", "day"], values="recall").round(3).to_string())
print(pd.DataFrame(rk).round(3).to_string())
