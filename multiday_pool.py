"""
Multi-day 2017 -> 2018 pool (Round 6 Reviewer 1, major 1).

Source (2017, one draw of 20,000 cleaned rows per day, 100,000 total):
  Monday (benign), Tuesday (FTP/SSH-Patator), Wednesday (DoS family),
  Friday-afternoon DDoS, Friday-afternoon PortScan.
Target (2018, 33,334 cleaned rows per day, about 100,000 total), the days
whose attack families have a 2017 counterpart:
  Wed 14 Feb (FTP/SSH brute force), Tue 20 Feb (DDoS LOIC-HTTP; a systematic
  1-in-26 sample of the day file, extract_sample.py), Wed 21 Feb (DDoS HOIC,
  LOIC-UDP).

Cleaning is identical to raw_pool.py: strip column names, drop the 2017
duplicate 'Fwd Header Length.1', drop 2018 Protocol/Timestamp (timestamp kept
for replay) and the Tue-20 identifier columns, inf -> NaN, drop NaN rows,
drop exact-duplicate rows (features + binary label), then a per-day draw with
RandomState(SEED). StandardScaler is fit on the source draw only.
Cached in multiday_cache.npz (git-ignored).

get_pool(featureset) -> dict(X17,y17,fam17,day17,X18,y18,fam18,day18,ts18,...)
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from verified_pipeline import SEED
from raw_pool import TEN_2017

UP = os.environ.get("UPLOADS", "/mnt/user-data/uploads")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multiday_cache.npz")
SRC = {
    "mon17": "Monday-WorkingHours_pcap_ISCX.csv",
    "tue17": "Tuesday-WorkingHours_pcap_ISCX.csv",
    "wed17": "Wednesday-workingHours_pcap_ISCX.csv",
    "friddos17": "Friday-WorkingHours-Afternoon-DDos_pcap_ISCX.csv",
    "friscan17": "Friday-WorkingHours-Afternoon-PortScan_pcap_ISCX.csv",
}
TGT = {
    "wed14_18": "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "tue20_18": "Tuesday-20-02-2018_sample.csv",
    "wed21_18": "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
}
N_SRC, N_TGT = 20_000, 33_334
DROP_2017 = ["Fwd Header Length.1"]
DROP_2018_ID = ["Flow ID", "Src IP", "Src Port", "Dst IP"]


def _load(day, path, is18, rng, n):
    df = pd.read_csv(os.path.join(UP, path), low_memory=False, encoding="latin1")
    df.columns = [c.strip() for c in df.columns]
    if is18:
        df = df.drop(columns=[c for c in DROP_2018_ID if c in df.columns])
        ts = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
        df["__ts"] = ts.astype("int64")
        df.loc[ts.isna(), "__ts"] = np.nan
        df = df.drop(columns=["Protocol", "Timestamp"])
    else:
        df = df.drop(columns=[c for c in DROP_2017 if c in df.columns])
    feats = [c for c in df.columns if c not in ("Label", "__ts")]
    assert len(feats) == 77, (day, len(feats))
    fam = df["Label"].astype(str).str.strip()
    y = (fam.str.upper() != "BENIGN").astype(np.float32)
    X = df[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    parts = [X, y.rename("Label"), fam.rename("Family")]
    if is18:
        parts.append(df["__ts"].rename("__ts"))
    out = pd.concat(parts, axis=1).dropna()
    out = out[out["Family"] != "Label"]
    out = out.drop_duplicates(subset=feats + ["Label"]).reset_index(drop=True)
    n_clean = len(out)
    if len(out) > n:
        out = out.iloc[rng.choice(len(out), n, replace=False)].sort_index().reset_index(drop=True)
    print(f"  {day}: {n_clean:,} clean rows -> {len(out):,} drawn, attack {out['Label'].mean():.3f}", flush=True)
    return out, feats


def _build():
    rng = np.random.RandomState(SEED)
    s_parts, t_parts, f17, f18 = [], [], None, None
    for day, p in SRC.items():
        d, f = _load(day, p, False, rng, N_SRC)
        d["Day"] = day
        s_parts.append(d); f17 = f17 or f
    for day, p in TGT.items():
        d, f = _load(day, p, True, rng, N_TGT)
        d["Day"] = day
        t_parts.append(d); f18 = f18 or f
    s, t = pd.concat(s_parts, ignore_index=True), pd.concat(t_parts, ignore_index=True)
    scaler = StandardScaler().fit(s[f17].values)
    np.savez_compressed(
        CACHE,
        X17=scaler.transform(s[f17].values).astype(np.float32), y17=s["Label"].values.astype(np.float32),
        fam17=s["Family"].values.astype(str), day17=s["Day"].values.astype(str),
        X18=scaler.transform(t[f18].values).astype(np.float32), y18=t["Label"].values.astype(np.float32),
        fam18=t["Family"].values.astype(str), day18=t["Day"].values.astype(str),
        ts18=t["__ts"].values.astype(np.int64), f17=np.array(f17), f18=np.array(f18))


def get_pool(featureset="77"):
    if not os.path.exists(CACHE):
        _build()
    z = np.load(CACHE, allow_pickle=True)
    f17 = list(z["f17"])
    idx = list(range(77)) if featureset == "77" else [f17.index(n) for n in TEN_2017]
    return dict(X17=z["X17"][:, idx], y17=z["y17"], fam17=z["fam17"], day17=z["day17"],
                X18=z["X18"][:, idx], y18=z["y18"], fam18=z["fam18"], day18=z["day18"],
                ts18=z["ts18"], features=[f17[i] for i in idx], idx=idx)


if __name__ == "__main__":
    p = get_pool("77")
    print("source", p["X17"].shape, f"attack {p['y17'].mean():.3f}", dict(zip(*np.unique(p["fam17"], return_counts=True))))
    print("target", p["X18"].shape, f"attack {p['y18'].mean():.3f}", dict(zip(*np.unique(p["fam18"], return_counts=True))))
    f18 = list(np.load(CACHE, allow_pickle=True)["f18"])
    print("column order agrees on first 5:", list(zip(p["features"][:5], f18[:5])))
