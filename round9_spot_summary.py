import json, numpy as np, scipy.stats as st, collections
import sys; d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "round9_spot.json"))
def tci(v):
    v = np.array(v, float); m = v.mean(); n = len(v)
    h = st.t.ppf(.975, n - 1) * v.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan"); return m, m - h, m + h
print("== replay: recall / precision / alert rate (mean over seeds of per-seed means; CI over 3 seeds)")
agg = collections.defaultdict(lambda: collections.defaultdict(list))
for k, rows in d["replay"].items():
    for r in rows: agg[k][r["seed"]].append(r)
for k in sorted(agg):
    ps = lambda f: [np.mean([r[f] for r in v]) for v in agg[k].values()]
    rec = tci(ps("recall")); print(f"{k:55s} recall {rec[0]:.3f} [{rec[1]:.3f},{rec[2]:.3f}] prec {np.mean(ps('precision')):.3f} alert {np.mean(ps('alert_rate')):.4f}")
print("== decoy: recall on real attacks")
agg = collections.defaultdict(lambda: collections.defaultdict(list))
for k, rows in d["decoy"].items():
    for r in rows: agg[k][r["seed"]].append(r["recall"])
for k in sorted(agg):
    v = [np.mean(x) for x in agg[k].values()]; c = tci(v); print(f"{k:35s} {c[0]:.3f} [{c[1]:.3f},{c[2]:.3f}]")
