import json, numpy as np, pandas as pd
rows = []
for pool in ("multi77", "raw77"):
    d = json.load(open(f"round7_dose_{pool}_raw.json"))
    for key, recs in d.items():
        day, sg = key.split("|")
        df = pd.DataFrame(recs)
        rows.append(dict(pool=pool, day=day, sigma=float(sg), **{c: float(df[c].mean()) for c in df.columns if c != "seed"}))
out = pd.DataFrame(rows); out["aqt_gain"] = out.aqt_macro - out.base_macro; out["budget_gain"] = out.budget_macro - out.base_macro
out.to_csv("round7_dose_summary.csv", index=False)
pd.set_option("display.width", 220)
print(out[["pool", "day", "sigma", "auroc", "tpr1", "base_macro", "aqt_macro", "budget_macro", "aqt_af1", "budget_af1"]].round(3).to_string(index=False))
