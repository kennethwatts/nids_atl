"""Round 9: q' contamination sweep (c up to 20%) and the contamination check, per day. Reads round9_robust_*_raw.json."""
import json, numpy as np
rows=[]
for pool in ("raw77","multi77"):
    d=json.load(open(f"round9_robust_{pool}_raw.json"))["robust"]
    for day,seeds in d.items():
        recs=[r for sd in seeds.values() for r in sd]
        for est in ("q99","qadj5","qadj5_chk"):
            for k in (500,1000,5000):
                for c in (0.0,0.01,0.05,0.10,0.20):
                    key=f"{est}_k{k}_c{c}"
                    rec=np.mean([r[key]["recall"] for r in recs]); pre=np.mean([r[key]["precision"] for r in recs])
                    rows.append((pool,day,est,k,c,round(rec,3),round(pre,3)))
import csv
with open("round9_robust_summary.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow("pool day est k c recall precision".split()); w.writerows(rows)
for r in rows:
    if r[3]==500 and r[2] in("qadj5","qadj5_chk","q99"): print(*r)
