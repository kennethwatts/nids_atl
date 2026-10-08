"""Second dataset family (Round 8, R1-3): NetFlow-v2 pair.

Source: NF-UNSW-NB15-v2 (UNSW-NB15 attacks, 2.39M flows, 4.0% attack).
Target: NF-CSE-CIC-IDS2018-v2 (150k-row order-preserving systematic sample made with
extract_sample.py / extract_sample_parquet.py, 11.9% attack).
Both share the 43-column NetFlow-v2 schema. We drop the IP addresses, both L4 ports and the
Label/Attack columns (37 features remain), apply a signed log1p (NetFlow counters are heavy
tailed), drop inf/NaN rows and exact duplicates (features + label), draw N rows per domain with
RandomState(SEED) at the natural class ratio, and fit StandardScaler on the source only.
Binary label = the Label column.  get_pool() -> dict(X17=source X, y17, X18=target X, y18).
Cached in nf_cache.npz (git-ignored).
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from verified_pipeline import SEED

UP = os.environ.get("NF_DIR", "/mnt/user-data/uploads")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nf_cache.npz")
SRC = os.environ.get("NF_SRC", os.path.join(UP, "NF-UNSW-NB15-v2.csv"))
TGT = os.environ.get("NF_TGT", os.path.join(UP, "NF-CSE-CIC-IDS2018-v2_sample.csv"))
DROP = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT", "Attack"]
N = 50_000


def _load(path, rng):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP if c in df.columns])
    df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
    df = df.iloc[np.sort(rng.choice(len(df), min(N, len(df)), replace=False))]
    y = df.pop("Label").values.astype(np.int64)
    X = df.values.astype(np.float64)
    return np.sign(X) * np.log1p(np.abs(X)), y, list(df.columns)


def get_pool(featureset=None):
    if os.path.exists(CACHE):
        z = np.load(CACHE); return {k: z[k] for k in z.files}
    rng = np.random.RandomState(SEED)
    Xs, ys, cols = _load(SRC, rng); Xt, yt, cols_t = _load(TGT, rng)
    assert cols == cols_t, set(cols) ^ set(cols_t)
    sc = StandardScaler().fit(Xs)
    out = dict(X17=sc.transform(Xs).astype(np.float32), y17=ys, X18=sc.transform(Xt).astype(np.float32), y18=yt)
    np.savez_compressed(CACHE, **out)
    return out


if __name__ == "__main__":
    p = get_pool()
    for k in ("17", "18"):
        print(k, p["X" + k].shape, "attack share", p["y" + k].mean().round(4), "max|z|", np.abs(p["X" + k]).max().round(1))
