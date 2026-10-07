"""20-seed gate diagnostic on paper10: oracle reset vs AQT, gated oracle, random 51.8% label mask."""
import glob, json
import numpy as np
from scipy import stats
d = {}
for f in sorted(glob.glob("round7_gate_paper10_shard*.json")): d.update(json.load(open(f)))
per = {}
for k, rec in d.items():
    s = int(k.split("_")[0]); per.setdefault(s, []).append({v: m["20000"]["macro_f1"] for v, m in rec.items()})
S = sorted(per); print("seeds", len(S))
avg = {s: {v: np.mean([u[v] for u in per[s]]) for v in per[s][0]} for s in S}
vars_ = list(avg[S[0]])
def ci(x):
    x = np.array(x); m = x.mean(); h = stats.t.ppf(.975, len(x)-1) * x.std(ddof=1) / np.sqrt(len(x)); return m, m-h, m+h
for v in vars_:
    if v == "aqt": continue
    m, lo, hi = ci([avg[s][v] - avg[s]["aqt"] for s in S])
    print(f"{v:22s} - aqt: {m:+.4f} [{lo:+.4f},{hi:+.4f}] pos {sum(avg[s][v] > avg[s]['aqt'] for s in S)}/{len(S)}")
print("levels", {v: round(float(np.mean([avg[s][v] for s in S])), 4) for v in vars_})
