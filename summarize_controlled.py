"""Writes controlled_feature_rerun_summary.csv from the controlled_raw{77,10}
shard files (macro-F1, attack-F1 and recall at n=20,000, with cluster-bootstrap
95% CIs over pretraining seeds and streams, plus paired differences)."""
import pandas as pd
from summarize_grid import table, diff

rows = []
for pool in ("raw10", "raw77"):
    for metric in ("macro_f1", "attack_f1", "recall"):
        t = table(f"controlled_{pool}", 2, "20000", metric)
        t.insert(0, "metric", metric); t.insert(0, "pool", pool)
        rows.append(t)
pd.concat(rows).to_csv("controlled_feature_rerun_summary.csv", index=False)
drows = []
for pool in ("raw10", "raw77"):
    for a, b in [("aqt_logit", "base"), ("aqt_logit", "aqt_prob"), ("oracle_logit", "aqt_logit"),
                 ("pseudo_logit", "aqt_logit"), ("oracle_logit", "pseudo_logit")]:
        d = diff(f"controlled_{pool}", 2, a, b); d["pool"] = pool; d["metric"] = "macro_f1"
        drows.append(d)
pd.DataFrame(drows).to_csv("controlled_feature_rerun_diffs.csv", index=False)
print("written")
