"""Summary of seed_headline.py: 5 pretraining seeds x 6 streams per pool, at
n in {1000, 5000, 10000, 20000}. Writes seed_headline_summary.csv (mean, cluster
bootstrap 95% CI over seeds then streams, min/max over seed means) and
seed_headline_diffs.csv (paired differences with CIs)."""
import pandas as pd
import summarize_grid
summarize_grid.REPS = 400  # many tables; 400 cluster-bootstrap reps keep CI endpoints stable to ~0.001
from summarize_grid import table, diff

parts, drows = [], []
for pool in ("paper10", "paper10_nodedup"):
    for n in ("1000", "5000", "10000", "20000"):
        for metric in ("macro_f1", "attack_f1", "recall", "precision", "alert_rate"):
            t = table(f"seedhead_{pool}", 2, n, metric)
            t.insert(0, "metric", metric); t.insert(0, "n", n); t.insert(0, "pool", pool)
            parts.append(t)
    for a, b in [("aqt", "base"), ("oracle_reset", "base"), ("pseudo_reset", "base"), ("unfreeze", "base"),
                 ("pseudo_reset", "aqt"), ("oracle_reset", "aqt"), ("pseudo_reset", "oracle_reset")]:
        for n in ("5000", "10000", "20000"):
            for metric in ("macro_f1", "attack_f1"):
                d = diff(f"seedhead_{pool}", 2, a, b, n, metric); d["pool"] = pool; d["n"] = n; d["metric"] = metric
                d["seed_diffs"] = str(d["seed_diffs"])
                drows.append(d)
pd.concat(parts).to_csv("seed_headline_summary.csv", index=False)
pd.DataFrame(drows).to_csv("seed_headline_diffs.csv", index=False)
print("written")
