"""Summarize time_ordered_finetune_raw.json (replay with fine-tuned variants)."""
import json, pandas as pd
d = json.load(open("time_ordered_finetune_raw.json")); rows = []
for o, vs in d.items():
    for v, recs in vs.items():
        for r in recs:
            rows.append(dict(order=o, variant=v, **r))
g = pd.DataFrame(rows).groupby(["order", "variant"])[["recall", "precision", "attack_f1", "alert_rate"]].mean()
g.to_csv("time_ordered_finetune_summary.csv"); print(g.round(3))
