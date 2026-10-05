"""Summary of pseudo_persistent_ablation.py (paper 10-feature data, pretraining
seed 42, 30 paired streams; CIs resample streams only because there is one
seed). Writes pseudo_persistent_summary.csv with macro-F1, attack-F1, recall and
the paired difference to AQT-alone for every variant."""
import pandas as pd
from summarize_grid import table, diff

parts = []
for metric in ("macro_f1", "attack_f1", "recall"):
    t = table("pseudo_persistent", 2, "20000", metric)
    t.insert(0, "metric", metric)
    parts.append(t)
pd.concat(parts).to_csv("pseudo_persistent_summary.csv", index=False)
rows = []
for v in table("pseudo_persistent", 2, "20000", "macro_f1")["variant"]:
    if v == "aqt":
        continue
    d = diff("pseudo_persistent", 2, v, "aqt"); d.pop("seed_diffs"); d.pop("all_seeds_positive")
    rows.append(d)
pd.DataFrame(rows).to_csv("pseudo_persistent_diffs_vs_aqt.csv", index=False)
print("written")
