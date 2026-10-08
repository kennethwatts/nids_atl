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

# paired decoy-vs-none differences per segment, streams as the unit (3 seeds x 8 streams)
print("recovery differences (decoy - none), 95% t-CI over 24 streams")
for rule in ("aqt","budget"):
    for seg in ("during","0-1000","1000-2000","2000+"):
        a={(k.split("_")[0],k.split("_")[1]):v for k,v in r.items() if k.split("_",4)[2]=="decoy" and k.split("_",4)[3]==rule and k.split("_",4)[4]==seg}
        b={(k.split("_")[0],k.split("_")[1]):v for k,v in r.items() if k.split("_",4)[2]=="none" and k.split("_",4)[3]==rule and k.split("_",4)[4]==seg}
        ks=[k for k in a if a[k] is not None and b.get(k) is not None]
        dd=np.array([a[k]-b[k] for k in ks]); h=st.t.ppf(.975,len(dd)-1)*dd.std(ddof=1)/np.sqrt(len(dd))
        print(rule,seg,len(dd),round(dd.mean(),3),round(dd.mean()-h,3),round(dd.mean()+h,3))
