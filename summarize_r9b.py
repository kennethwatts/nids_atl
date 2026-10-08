"""Summaries for round9_extras.py: prevalence sweep (20 seeds) and decoy recovery."""
import json, glob, numpy as np, csv
from scipy import stats as st
d={}
for f in glob.glob("round9_prev_shard*.json"): d.update(json.load(open(f)))
rows=[]
for prev in (0.005,0.01,0.02,0.05,0.20):
    for rule in ("q0.99","q_matched"):
        by={}
        for k,r in d.items():
            sd,j,p=k.split("_")
            if float(p)==prev: by.setdefault(sd,[]).append(r[rule]["attack_f1"])
        x=np.array([np.mean(v) for v in by.values()]); n=len(x); h=st.t.ppf(.975,n-1)*x.std(ddof=1)/np.sqrt(n)
        rows.append((prev,rule,n,round(x.mean(),3),round(x.mean()-h,3),round(x.mean()+h,3))); print(*rows[-1])
csv.writer(open("round9_prev_summary.csv","w")).writerows([("prev","rule","seeds","attackF1","lo","hi")]+rows)
r=json.load(open("round9_recover.json")); print("recovery")
for rule in ("aqt","budget"):
    for tag in ("decoy","none"):
        out=[]
        for seg in ("during","0-1000","1000-2000","2000+"):
            by={}
            for k,v in r.items():
                sd,j,t,ru,sg=k.split("_",4)
                if t==tag and ru==rule and sg==seg and v is not None: by.setdefault(sd,[]).append(v)
            out.append(round(float(np.mean([np.mean(v) for v in by.values()])),3))
        print(rule,tag,out)
