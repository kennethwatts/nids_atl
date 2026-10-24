"""Round 10 (mock-review R1.1/R2.4/R4.4): frozen R-precision of a tree ensemble (HistGradientBoosting, source-trained,
no adaptation) against the MLP on the five pools. 5 seeds x 4 streams of 20,000 flows at 1% prevalence.
R-precision = recall in the top 1% of scores; also AUROC and TPR@1%FPR. usage: python3 round10_trees.py -> round10_trees.json"""
import json, numpy as np, torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from verified_pipeline import pretrain_source_model, make_cold_start_stream
from grid_runner import load_pool
from threshold_baselines import scores_of
N = 20000; out = {}
def mets(s, y):
    k = int(round(0.01 * len(s))); top = np.argsort(-s, kind="stable")[:k]
    fpr, tpr, _ = roc_curve(y, s); return dict(auroc=float(roc_auc_score(y, s)), rprec=float(y[top].sum() / y.sum()),
                                               tpr1=float(np.interp(0.01, fpr, tpr)))
for pool in ("paper10", "raw77", "multi77", "multi10", "nf"):
    X17, y17, X18, y18 = load_pool(pool); dim = X17.shape[1]; space = "prob" if pool == "paper10" else "logit"
    for seed in (42, 123, 456, 789, 2024):
        torch.manual_seed(seed); m = pretrain_source_model(X17, y17, dim)
        gb = HistGradientBoostingClassifier(random_state=seed).fit(X17, y17)
        for j in range(4):
            rng = np.random.RandomState(seed * 1000 + j); Xs, ys = make_cold_start_stream(X18, y18, N, 0.01, rng)
            sm = scores_of(m, Xs, space); sg = gb.decision_function(Xs)
            out.setdefault(pool, []).append(dict(seed=seed, j=j, mlp=mets(sm, ys), gb=mets(sg, ys)))
    a = out[pool]; f = lambda mod, k: np.mean([r[mod][k] for r in a])
    print(pool, {mod: {k: round(f(mod, k), 3) for k in ("auroc", "rprec", "tpr1")} for mod in ("mlp", "gb")}, flush=True)
    json.dump(out, open("round10_trees.json", "w"))
