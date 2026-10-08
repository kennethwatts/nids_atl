"""Round 8 R1-1/R1-2: R-precision (recall in the top 1% of a 1%-prevalence stream) next to TPR at 1% FPR,
for every pool and, for the multi-day pair, every 2018 day separately. Frozen model, 3 seeds, 4 i.i.d. streams.
Also the operating FPR of the top-1% rule, pi(1-R)/(1-pi).
usage: python3 round8_frozen.py   -> round8_rprec.csv"""
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from verified_pipeline import pretrain_source_model, make_cold_start_stream
from threshold_baselines import scores_of
from grid_runner import load_pool
torch.set_num_threads(1)
SEEDS = [42, 123, 456]
rows = []
def units(pool):
    if pool == "multi77":
        from multiday_pool import get_pool
        p = get_pool("77")
        for d in np.unique(p["day18"]):
            m = p["day18"] == d; yield d, p["X17"], p["y17"], p["X18"][m], p["y18"][m]
        yield "pooled", p["X17"], p["y17"], p["X18"], p["y18"]
    else:
        X17, y17, X18, y18 = load_pool(pool); yield "all", X17, y17, X18, y18
for pool in ("paper10", "raw77", "multi77", "multi10", "nf"):
    for day, X17, y17, X18, y18 in units(pool):
        for seed in SEEDS:
            torch.manual_seed(seed); m = pretrain_source_model(X17, y17, X17.shape[1])
            s_all = scores_of(m, X18, "logit")
            b = s_all[y18 == 0]; thr = np.quantile(b, 0.99)
            for rep in range(4):
                rng = np.random.RandomState(seed * 1000 + rep)
                ix, yy = make_cold_start_stream(np.arange(len(s_all)).reshape(-1, 1).astype(np.float32), y18, 20000, 0.01, rng)
                ix = ix.ravel().astype(int); s = s_all[ix]; y = yy
                k = int(round(0.01 * len(s))); top = np.argsort(-s)[:k]
                R = float(y[top].sum() / max(y.sum(), 1)); pi = float(y.mean())
                rows.append(dict(pool=pool, day=day, seed=seed, rep=rep, auroc=float(roc_auc_score(y, s)),
                                 tpr1=float((s[y == 1] > thr).mean()), rprec=R, op_fpr=pi * (1 - R) / (1 - pi)))
        print("done", pool, day, flush=True)
df = pd.DataFrame(rows)
out = df.groupby(["pool", "day"])[["auroc", "tpr1", "rprec", "op_fpr"]].mean().round(4).reset_index()
out.to_csv("round8_rprec.csv", index=False); print(out.to_string())
