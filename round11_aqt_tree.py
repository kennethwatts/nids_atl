"""Round 11 (advisor re-review item 4): measured AQT attack-F1 on the tree ensemble's scores vs the MLP's, same streams as
round10_trees.py (5 seeds x 4 streams of 20,000 flows, 1% prevalence, q=0.99, W=500, pre-seeded with 500 random source scores).
usage: python3 round11_aqt_tree.py -> round11_aqt_tree.json"""
import json, numpy as np, torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from verified_pipeline import pretrain_source_model, make_cold_start_stream
from grid_runner import load_pool
from threshold_baselines import scores_of, aqt_preds
N = 20000; out = {}
for pool in ("paper10", "raw77", "multi77", "multi10", "nf"):
    X17, y17, X18, y18 = load_pool(pool); dim = X17.shape[1]; space = "prob" if pool == "paper10" else "logit"
    for seed in (42, 123, 456, 789, 2024):
        torch.manual_seed(seed); m = pretrain_source_model(X17, y17, dim)
        gb = HistGradientBoostingClassifier(random_state=seed).fit(X17, y17)
        rs = np.random.RandomState(seed); pi = rs.choice(len(X17), 500, replace=False)
        pre_m = scores_of(m, X17[pi], space); pre_g = gb.decision_function(X17[pi])
        for j in range(4):
            rng = np.random.RandomState(seed * 1000 + j); Xs, ys = make_cold_start_stream(X18, y18, N, 0.01, rng)
            r = {}
            for name, s, pre, d0 in (("mlp", scores_of(m, Xs, space), pre_m, 0.5 if space == "prob" else 0.0), ("gb", gb.decision_function(Xs), pre_g, 0.0)):
                p = aqt_preds(s, pre, 500, 0.99, d0)
                r[name] = dict(af1=float(f1_score(ys, p, zero_division=0)), alert=float(p.mean()))
            out.setdefault(pool, []).append(dict(seed=seed, j=j, **r))
    a = out[pool]; print(pool, {k: round(float(np.mean([x[k]["af1"] for x in a])), 3) for k in ("mlp", "gb")}, flush=True)
    json.dump(out, open("round11_aqt_tree.json", "w"))
