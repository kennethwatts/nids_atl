"""Summaries for the 20-seed grid: per-seed means, then a t-interval ACROSS
SEEDS (seeds are the independent units; 20 clusters make the t-interval
reliable where a 5-cluster bootstrap is not). usage: python3 summarize_multiseed.py <pool>"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
from summarize_grid import to_frame

pool = sys.argv[1]
metric = sys.argv[2] if len(sys.argv) > 2 else "macro_f1"
df = to_frame(f"multiseed_{pool}", 2, "20000", metric)
w = df.pivot_table(index=["seed", "stream"], columns="variant", values="value").reset_index()
per_seed = w.drop(columns="stream").groupby("seed").mean(numeric_only=True)
n = len(per_seed)

def tci(x):
    x = np.asarray(x, float); m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(0.975, len(x) - 1) * se
    return m, m - h, m + h

rows = []
for v in per_seed.columns:
    m, lo, hi = tci(per_seed[v])
    rows.append(dict(kind="level", a=v, b="", mean=m, lo=lo, hi=hi, seed_min=per_seed[v].min(),
                     seed_max=per_seed[v].max(), seeds_positive=np.nan))
ref = [("aqt", "base"), ("oracle_reset", "base"), ("pseudo_reset", "base"),
       ("oracle_reset", "aqt"), ("pseudo_reset", "aqt")]
ref += [(v, "aqt") for v in per_seed.columns if v.startswith(("oracle_adam", "oracle_sgd", "pseudo_adam"))]
for a, b in ref:
    if a in per_seed and b in per_seed:
        d = per_seed[a] - per_seed[b]
        m, lo, hi = tci(d)
        rows.append(dict(kind="diff", a=a, b=b, mean=m, lo=lo, hi=hi, seed_min=d.min(), seed_max=d.max(),
                         seeds_positive=int((d > 0).sum())))
out = pd.DataFrame(rows); out["n_seeds"] = n
out.to_csv(f"multiseed_{pool}_summary.csv" if metric == "macro_f1" else f"multiseed_{pool}_{metric}_summary.csv", index=False)
pd.set_option("display.width", 220)
print(out.round(4).to_string())
