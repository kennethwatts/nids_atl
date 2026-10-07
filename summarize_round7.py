"""Round 7 summaries. Per-seed macro-F1 / attack-F1 at n=20000, then t-intervals across seeds
(seeds are the independent unit). Merges the new Round 7 shards with the earlier 20-seed
shards for the same pool (identical seeds and streams) and checks that the repeated
`base` and `aqt` variants agree.
usage: python3 summarize_round7.py <cells|tune|labels> <pool> [metric]"""
import glob, json, sys
import numpy as np, pandas as pd
from scipy import stats

mode, pool = sys.argv[1], sys.argv[2]
metric = sys.argv[3] if len(sys.argv) > 3 else "macro_f1"
OLD = {"paper10": "multiseed_paper10", "multi77": "multiday_multi77", "multi10": "multiday_multi10"}


def load(prefix):
    d = {}
    for f in sorted(glob.glob(f"{prefix}_shard*.json")):
        d.update(json.load(open(f)))
    return d


new = load(f"round7_{mode}_{pool}")
old = load(OLD[pool]) if (mode == "cells" and pool in OLD) else {}
units, check = {}, []
for key, rec in new.items():
    seed = int(key.split("_")[0]); r = {v: m["20000"][metric] for v, m in rec.items()}
    if key in old:
        for v in ("base", "aqt"):
            if v in old[key] and v in r:
                check.append(abs(old[key][v]["20000"][metric] - r[v]))
        for v, m in old[key].items():
            r.setdefault(v, m["20000"][metric])
    units[key] = r
# several streams per seed (paper pair, first five seeds): average within seed, as in summarize_multiseed.py
per = pd.DataFrame(units).T; per["seed"] = [int(k.split("_")[0]) for k in per.index]
rows = {sd: g.drop(columns="seed").mean().to_dict() for sd, g in per.groupby("seed")}
if check:
    print(f"paired check vs earlier run (base, aqt): max |diff| = {max(check):.2e} over {len(check)} comparisons")
df = pd.DataFrame(rows).T.sort_index(); n = len(df)


def tci(x, alpha=0.05):
    x = np.asarray(x, float); m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x)); h = stats.t.ppf(1 - alpha / 2, len(x) - 1) * se
    return m, m - h, m + h


out = []
for v in df.columns:
    m, lo, hi = tci(df[v]); out.append(dict(kind="level", a=v, b="", mean=m, lo=lo, hi=hi, seeds_positive=np.nan))
pairs = [(a, "aqt") for a in df.columns if a not in ("aqt", "base")] + [("aqt", "base"), ("budget_1pct", "aqt"), ("budget_1pct", "base")]
ncomp = len([p for p in pairs if p[0] in df and p[1] in df])
for a, b in pairs:
    if a in df and b in df:
        d = (df[a] - df[b]).dropna(); m, lo, hi = tci(d); _, blo, bhi = tci(d, 0.05 / max(ncomp, 1))
        out.append(dict(kind="diff", a=a, b=b, mean=m, lo=lo, hi=hi, seeds_positive=int((d > 0).sum()),
                        bonf_excludes_zero=bool(blo > 0 or bhi < 0)))
res = pd.DataFrame(out); res["n_seeds"] = n; res["n_comparisons"] = ncomp
res.to_csv(f"round7_{mode}_{pool}_{metric}_summary.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(4).to_string())
