"""Round 9: pseudo-label 300x (grid edge) and the NF label-budget curve. Reads round7_r9tune_*/round7_r9_* shards.
Prints macro-F1 diff vs AQT (mean, 95% t-CI over seeds) and writes round9_summary.csv."""
import json, glob, csv, numpy as np
from scipy import stats
def load(mode, pool):
    d={}
    for f in glob.glob(f"round7_{mode}_{pool}_shard*.json"): d.update(json.load(open(f)))
    return d
def diff(d, v, metric="macro_f1"):
    x=[]
    for k,r in d.items():
        if v in r and "aqt" in r: x.append(r[v]["20000"][metric]-r["aqt"]["20000"][metric])
    x=np.array(x); n=len(x); m=x.mean(); h=stats.t.ppf(.975,n-1)*x.std(ddof=1)/np.sqrt(n)
    return m,m-h,m+h,n,int((x>0).sum())
rows=[]
for mode,pool,vs in [("r9tune","nf",["pseudo_adam_300x"]),("r9tune","multi10",["pseudo_adam_300x"]),
                     ("r9","multi10",["pseudo_adam_300x"]),
                     ("r9","nf",["pseudo_adam_300x"]+[f"lab{f}_adam_{m}x" for m in (3,10) for f in (0.01,0.05,0.1,0.25,1.0)])]:
    d=load(mode,pool)
    if not d: continue
    for v in vs:
        try: m,lo,hi,n,pos=diff(d,v)
        except Exception as e: print("err",e); continue
        rows.append((mode,pool,v,round(m,4),round(lo,4),round(hi,4),n,pos)); print(*rows[-1])
csv.writer(open("round9_summary.csv","w")).writerows([("mode","pool","variant","mean","lo","hi","n","pos")]+rows)
