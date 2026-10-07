"""Seed-averaged replacement for Table I: macro-F1 at n in {100,1000,5000,10000,20000} and attack-F1 at 20000,
20 pretraining seeds (per-seed mean over streams), 95% t-interval across seeds. Deduplicated paper pair."""
import glob, json, numpy as np, pandas as pd
from scipy import stats
d = {}
for f in sorted(glob.glob("multiseed_paper10_shard*.json")): d.update(json.load(open(f)))
vars_ = ["base", "aqt", "oracle_reset", "pseudo_reset"]; ns = ["100", "1000", "5000", "10000", "20000"]
rows = []
for v in vars_:
    for n in ns + ["af1_20000"]:
        per = {}
        for key, rec in d.items():
            seed = int(key.split("_")[0])
            val = rec[v]["20000"]["attack_f1"] if n == "af1_20000" else rec[v][n]["macro_f1"]
            per.setdefault(seed, []).append(val)
        x = np.array([np.mean(per[s]) for s in sorted(per)]); m = x.mean(); h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        rows.append(dict(variant=v, n=n, mean=m, lo=m - h, hi=m + h, n_seeds=len(x)))
out = pd.DataFrame(rows); out.to_csv("round7_table1_seedavg.csv", index=False)
print(out.pivot(index="variant", columns="n", values="mean").round(3).to_string())
print(out[out.n == "20000"].round(3).to_string())
