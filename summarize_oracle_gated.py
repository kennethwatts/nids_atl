"""Summary of oracle_gated_ablation.py (paper 10-feature data, seed 42, 30
paired streams, reset Adam): macro-F1 / attack-F1 / output bias at n=20,000 and
paired differences with normal-approximation 95% CIs over streams."""
import json
import numpy as np
import pandas as pd

rec = {}
for k in (0, 1):
    rec.update(json.load(open(f"oracle_gated_shard{k}.json")))
rows = []
a = np.array([r["aqt"]["20000"]["macro_f1"] for r in rec.values()])
for v in ["aqt", "oracle_reset", "oracle_gated_reset", "pseudo_reset"]:
    mf = np.array([r[v]["20000"]["macro_f1"] for r in rec.values()])
    af = np.array([r[v]["20000"]["attack_f1"] for r in rec.values()])
    d = mf - a; se = d.std(ddof=1) / np.sqrt(len(d)) if v != "aqt" else 0.0
    rows.append({"variant": v, "macro_f1": mf.mean(), "attack_f1": af.mean(),
                 "updates": np.mean([r[v]["n_updates"] for r in rec.values()]),
                 "bias_at_20000": np.mean([r[v]["trace"][-1]["bias"] for r in rec.values()]) if "trace" in rec[next(iter(rec))][v] else np.nan,
                 "diff_vs_aqt": d.mean(), "diff_ci_lo": d.mean() - 1.96 * se, "diff_ci_hi": d.mean() + 1.96 * se})
pd.DataFrame(rows).to_csv("oracle_gated_summary.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))
