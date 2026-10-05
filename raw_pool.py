"""
Cached raw-file pool (2017 source, 2018 target) with family labels, used by
the Round 5 experiments.

Cleaning is identical to all_features_rerun.py: strip column names, drop
2017 'Fwd Header Length.1' and 2018 'Protocol'/'Timestamp' (Timestamp is kept
separately here for time-ordered replay), inf -> NaN, drop NaN rows, drop
exact-duplicate rows (features + binary label), subsample to 100,000 rows per
domain with RandomState(SEED). StandardScaler is fit on the source subsample
only. The cleaned arrays are cached in pool_cache.npz (git-ignored) because
parsing the two raw CSVs takes ~2 minutes.

get_pool(featureset) returns a dict with X17, y17, fam17, X18, y18, fam18,
ts18 (target timestamps, int64 ns, for replay), feature names, and the
column indices used. featureset is '77' or '10' (the paper's ten pipeline
features, mapped into the raw columns by name).
"""

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from verified_pipeline import SEED
from all_features_rerun import SRC_2017, TGT_2018, DROP_2017, POOL_SIZE

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pool_cache.npz")

# The paper's 10 pipeline features, named as in the 2017 and 2018 raw exports.
TEN_2017 = ["Bwd Packet Length Std", "Packet Length Variance", "Average Packet Size",
            "Packet Length Std", "Avg Bwd Segment Size", "Bwd Packet Length Mean",
            "Subflow Bwd Bytes", "Packet Length Mean", "Total Length of Fwd Packets",
            "Subflow Fwd Bytes"]
TEN_2018 = ["Bwd Pkt Len Std", "Pkt Len Var", "Pkt Size Avg", "Pkt Len Std",
            "Bwd Seg Size Avg", "Bwd Pkt Len Mean", "Subflow Bwd Byts", "Pkt Len Mean",
            "TotLen Fwd Pkts", "Subflow Fwd Byts"]


def _build():
    df17 = pd.read_csv(SRC_2017, low_memory=False)
    df17.columns = [c.strip() for c in df17.columns]
    df17 = df17.drop(columns=[c for c in DROP_2017 if c in df17.columns])
    df18 = pd.read_csv(TGT_2018, low_memory=False)
    df18.columns = [c.strip() for c in df18.columns]
    ts = pd.to_datetime(df18["Timestamp"], dayfirst=True, errors="coerce")
    df18["__ts"] = ts.astype("int64")  # NaT -> min int64, dropped below via dropna on __ts
    df18.loc[ts.isna(), "__ts"] = np.nan
    df18 = df18.drop(columns=["Protocol", "Timestamp"])

    f17 = [c for c in df17.columns if c != "Label"]
    f18 = [c for c in df18.columns if c not in ("Label", "__ts")]
    assert len(f17) == len(f18) == 77

    def prep(df, cols, with_ts):
        fam = df["Label"].astype(str).str.strip()
        y = (fam.str.upper() != "BENIGN").astype(np.float32)
        X = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        parts = [X, y.rename("Label"), fam.rename("Family")]
        if with_ts:
            parts.append(df["__ts"].rename("__ts"))
        out = pd.concat(parts, axis=1).dropna()
        return out.drop_duplicates(subset=list(cols) + ["Label"]).reset_index(drop=True)

    d17, d18 = prep(df17, f17, False), prep(df18, f18, True)
    rng = np.random.RandomState(SEED)
    if len(d17) > POOL_SIZE:
        d17 = d17.iloc[rng.choice(len(d17), POOL_SIZE, replace=False)].reset_index(drop=True)
    if len(d18) > POOL_SIZE:
        d18 = d18.iloc[rng.choice(len(d18), POOL_SIZE, replace=False)].reset_index(drop=True)
    scaler = StandardScaler().fit(d17[f17].values)
    np.savez_compressed(
        CACHE,
        X17=scaler.transform(d17[f17].values).astype(np.float32),
        y17=d17["Label"].values.astype(np.float32),
        fam17=d17["Family"].values.astype(str),
        X18=scaler.transform(d18[f18].values).astype(np.float32),
        y18=d18["Label"].values.astype(np.float32),
        fam18=d18["Family"].values.astype(str),
        ts18=d18["__ts"].values.astype(np.int64),
        f17=np.array(f17), f18=np.array(f18),
    )


def get_pool(featureset="77"):
    if not os.path.exists(CACHE):
        _build()
    z = np.load(CACHE, allow_pickle=True)
    f17, f18 = list(z["f17"]), list(z["f18"])
    if featureset == "77":
        idx = list(range(77))
    elif featureset == "10":
        idx = [f17.index(n) for n in TEN_2017]
        assert [f18[i] for i in idx] == TEN_2018, "2017/2018 column mapping disagrees for the 10 pipeline features"
    else:
        raise ValueError(featureset)
    return dict(X17=z["X17"][:, idx], y17=z["y17"], fam17=z["fam17"],
                X18=z["X18"][:, idx], y18=z["y18"], fam18=z["fam18"], ts18=z["ts18"],
                features=[f17[i] for i in idx], idx=idx)


if __name__ == "__main__":
    for fs in ("77", "10"):
        p = get_pool(fs)
        print(fs, p["X17"].shape, p["X18"].shape, f"src attack {p['y17'].mean():.3f}, tgt attack {p['y18'].mean():.3f}")
    print("10-feature names:", get_pool("10")["features"])
    ts = get_pool("77")["ts18"]
    print("timestamp range:", pd.to_datetime(ts.min()), "->", pd.to_datetime(ts.max()))
