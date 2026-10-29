"""Round 12 (advisor stretch goal): classic CORAL (Sun, Feng and Saenko, AAAI 2016) on the frozen source MLP.
The target stream is re-colored to the source covariance before scoring: X' = X (Ct+I)^(-1/2) (Cs+I)^(1/2), with Cs from all
source rows and Ct from (a) the first 2,000 target flows of each stream (label-free, deployable) or (b) the whole target pool
(upper bound). Frozen model, q=0.99 AQT (W=500, pre-seeded with 500 source scores), 1% prevalence, 20,000-flow streams.
Reports AUROC, R-precision and AQT attack-F1 for none / coral2000 / coralpool, 5 seeds x 4 streams, same streams for all.
usage: python3 round12_coral.py [pool ...]  -> round12_coral_<pool>.json"""
import sys, json, numpy as np, torch
from sklearn.metrics import f1_score, roc_auc_score
from verified_pipeline import pretrain_source_model, make_cold_start_stream
from grid_runner import load_pool
from threshold_baselines import scores_of, aqt_preds
torch.set_num_threads(2)
N = 20000; SEEDS = (42, 123, 456, 789, 2024)


def msqrt(C, p):
    w, V = np.linalg.eigh(C); w = np.clip(w, 1e-8, None)
    return (V * w ** p) @ V.T


def cov(X):
    Xc = X - X.mean(0, keepdims=True)
    return Xc.T @ Xc / max(len(X) - 1, 1)


def coral(X, Cs_half, Ct):
    d = X.shape[1]
    return X @ msqrt(Ct + np.eye(d), -0.5) @ Cs_half


def metrics(s, y, pre, d0):
    k = int(round(0.01 * len(s))); top = np.argsort(-s)[:k]
    p = aqt_preds(s, pre, 500, 0.99, d0)
    return dict(auroc=float(roc_auc_score(y, s)), rprec=float(y[top].sum() / max(y.sum(), 1)),
                af1=float(f1_score(y, p, zero_division=0)), alert=float(p.mean()))


def run(pool):
    X17, y17, X18, y18 = load_pool(pool); dim = X17.shape[1]; space = "prob" if pool == "paper10" else "logit"
    d0 = 0.5 if space == "prob" else 0.0
    X17d = X17.astype(np.float64); Cs_half = msqrt(cov(X17d) + np.eye(dim), 0.5)
    Ct_pool = cov(X18.astype(np.float64)); out = []
    for seed in SEEDS:
        torch.manual_seed(seed); m = pretrain_source_model(X17, y17, dim)
        rs = np.random.RandomState(seed); pi = rs.choice(len(X17), 500, replace=False)
        pre = scores_of(m, X17[pi], space)
        for j in range(4):
            rng = np.random.RandomState(seed * 1000 + j); Xs, ys = make_cold_start_stream(X18, y18, N, 0.01, rng)
            Xs64 = Xs.astype(np.float64); r = dict(seed=seed, j=j)
            r["none"] = metrics(scores_of(m, Xs, space), ys, pre, d0)
            Xc = coral(Xs64, Cs_half, cov(Xs64[:2000])).astype(np.float32)
            r["coral2000"] = metrics(scores_of(m, Xc, space), ys, pre, d0)
            Xp = coral(Xs64, Cs_half, Ct_pool).astype(np.float32)
            r["coralpool"] = metrics(scores_of(m, Xp, space), ys, pre, d0)
            out.append(r)
        print(pool, seed, {k: round(float(np.mean([x[k]["rprec"] for x in out if x["seed"] == seed])), 3) for k in ("none", "coral2000", "coralpool")}, flush=True)
        json.dump(out, open(f"round12_coral_{pool}.json", "w"))
    return out


if __name__ == "__main__":
    for p in (sys.argv[1:] or ["paper10", "raw77", "multi77", "multi10", "nf"]):
        run(p)
