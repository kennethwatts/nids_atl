import json, numpy as np, pandas as pd
d = json.load(open("multiday_replay_raw.json")); rows = []
for key, seeds in d.items():
    day, order = key.split("|")
    recs = [r for s in seeds.values() for r in s]
    for rule in recs[0]:
        if rule.startswith("_"): continue
        rows.append(dict(day=day, order=order, rule=rule, **{m: np.mean([r[rule][m] for r in recs]) for m in ("recall", "precision", "attack_f1", "alert_rate")}))
    rows.append(dict(day=day, order=order, rule="_cluster", recall=np.mean([r["_cluster"]["densest5"] for r in recs]), precision=np.mean([r["_cluster"]["peak500"] for r in recs]), attack_f1=np.nan, alert_rate=np.nan))
df = pd.DataFrame(rows); df.to_csv("multiday_replay_summary.csv", index=False)
pd.set_option("display.width", 200)
print(df[df.rule.isin(["aqt_W500", "budget_1pct", "static_benign_500", "aqt_cap_benign_500", "aqt_W5000", "_cluster"])].round(3).to_string())
