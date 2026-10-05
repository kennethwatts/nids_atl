"""
Summaries for grid_runner results: per-variant means, across-seed ranges and
cluster-bootstrap confidence intervals (resample pretraining seeds, then
streams within each seed), plus paired differences between variants.

Seeds are treated as clusters, because streams that share a pretrained model
are not independent. With few seeds the bootstrap interval is wide on
purpose.
"""

import numpy as np
import pandas as pd

from grid_runner import load_all


def to_frame(name, nshards, n="20000", metric="macro_f1"):
    rec = load_all(name, nshards)
    rows = []
    for key, variants in rec.items():
        seed, j = key.split("_")
        for v, r in variants.items():
            rows.append({"seed": int(seed), "stream": int(j), "variant": v,
                         "value": r[n][metric]})
    return pd.DataFrame(rows)


def cluster_boot(df_wide, fn, reps=2000, seed=0):
    """df_wide: rows = (seed, stream) units, columns = variants. fn maps a
    wide frame to a scalar. Resamples seeds with replacement, then streams
    within each chosen seed."""
    rng = np.random.RandomState(seed)
    groups = {s: g for s, g in df_wide.groupby("seed")}
    seeds = list(groups)
    vals = []
    for _ in range(reps):
        pick = rng.choice(seeds, len(seeds), replace=True)
        parts = []
        for s in pick:
            g = groups[s]
            parts.append(g.iloc[rng.choice(len(g), len(g), replace=True)])
        vals.append(fn(pd.concat(parts)))
    return np.percentile(vals, [2.5, 97.5])


def table(name, nshards, n="20000", metric="macro_f1", variants=None):
    df = to_frame(name, nshards, n, metric)
    wide = df.pivot_table(index=["seed", "stream"], columns="variant", values="value").reset_index()
    out = []
    for v in (variants or [c for c in wide.columns if c not in ("seed", "stream")]):
        per_seed = wide.groupby("seed")[v].mean()
        lo, hi = cluster_boot(wide[["seed", v]], lambda d, v=v: d[v].mean())
        out.append({"variant": v, "mean": wide[v].mean(), "ci95_lo": lo, "ci95_hi": hi,
                    "seed_min": per_seed.min(), "seed_max": per_seed.max(), "units": len(wide)})
    return pd.DataFrame(out)


def diff(name, nshards, a, b, n="20000", metric="macro_f1"):
    df = to_frame(name, nshards, n, metric)
    wide = df.pivot_table(index=["seed", "stream"], columns="variant", values="value").reset_index()
    d = (wide[a] - wide[b])
    per_seed = d.groupby(wide["seed"]).mean()
    lo, hi = cluster_boot(wide[["seed", a, b]], lambda x: (x[a] - x[b]).mean())
    return {"a": a, "b": b, "mean_diff": float(d.mean()), "ci95_lo": float(lo), "ci95_hi": float(hi),
            "seed_diffs": [round(float(x), 4) for x in per_seed.values],
            "all_seeds_positive": bool((per_seed > 0).all())}
