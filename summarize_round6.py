"""Summaries for the Round 6 frozen-model experiments: decoy attackers, mitigation replay, q sweep, BBSE."""
import json
import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

def decoy(pool):
    d = json.load(open(f"round6_decoy_{pool}_raw.json")); rows = []
    for k, v in d.items():
        s, j, r, a = k.split("_")
        for rule in ("aqt", "budget_1pct"):
            rows.append(dict(seed=s, rate=float(r), attacker=a, rule=rule, recall=v[rule]["recall"], alert=v[rule]["alert_rate"], hit=v["surrogate_hit_rate"]))
    df = pd.DataFrame(rows)
    t = df.pivot_table(index=["rule", "attacker"], columns="rate", values="recall")
    t.to_csv(f"round6_decoy_{pool}_summary.csv")
    print(pool, "surrogate decoys in victim top-1%:", round(df.hit.mean(), 3)); print(t.round(3))

def mitig():
    d = json.load(open("round6_mitigations_replay_raw.json")); rows = []
    for key, seeds in d.items():
        fam, order = key.split("|")
        for seed, recs in seeds.items():
            for rec in recs:
                for r, m in rec.items():
                    rows.append(dict(fam=fam, order=order, seed=seed, rule=r, **m))
    df = pd.DataFrame(rows)
    g = df.groupby(["fam", "order", "rule"])[["recall", "precision", "attack_f1", "alert_rate"]].mean()
    g.to_csv("round6_mitigations_replay_summary.csv")
    print(g.loc["all"].round(3).to_string())

def qsweep():
    q = json.load(open("round6_qsweep_raw77_raw.json")); rows = []
    for k, v in q.items():
        s, j, p = k.split("_")
        for r, m in v.items():
            rows.append(dict(prev=float(p), rule=r, f1=m["attack_f1"], rec=m["recall"], alert=m["alert_rate"]))
    t = pd.DataFrame(rows).pivot_table(index="prev", columns="rule", values="f1")
    t.to_csv("round6_qsweep_raw77_summary.csv"); print(t.round(3))

def bbse():
    d = json.load(open("round6_bbse_raw.json")); rows = []
    for k, v in d.items():
        pool, seed, thr = k.split("|"); rows.append(dict(pool=pool, thr=thr, **v))
    t = pd.DataFrame(rows).groupby(["pool", "thr"])[["true_pi", "est"]].agg(["mean", "min", "max"])
    t.to_csv("round6_bbse_summary.csv"); print(t.round(3))

if __name__ == "__main__":
    for p in ("raw77", "paper10"):
        decoy(p)
    mitig(); qsweep(); bbse()
