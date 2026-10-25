"""Run A on the 77-feature pools (mock-review plan): does re-standardizing the target stream from its own unlabeled flows
raise frozen AUROC / R-precision? Variants: src (source stats), mean_std500 / med_mad500 (first 500 stream flows),
mean_stdall (whole stream, oracle). Frozen MLP from verified_pipeline (repo torch code). usage: python3 round11_runA.py POOL NSEEDS"""
import sys, json, numpy as np, torch
from scipy import stats
from sklearn.metrics import roc_auc_score
from verified_pipeline import pretrain_source_model, make_cold_start_stream
from grid_runner import load_pool
from threshold_baselines import scores_of
pool = sys.argv[1]; ns = int(sys.argv[2]); N = 20000
X17, y17, X18, y18 = load_pool(pool); dim = X17.shape[1]; space = "prob" if pool == "paper10" else "logit"
mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x, 0)), 0) + 1e-6
def rp(s, y):
    k = int(round(0.01 * len(s))); top = np.argsort(-s, kind="stable")[:k]; return y[top].sum() / y.sum()
seeds = [42, 123, 456, 789, 2024, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61][:ns]
rows = []
for seed in seeds:
    torch.manual_seed(seed); m = pretrain_source_model(X17, y17, dim)
    for j in range(4):
        rng = np.random.RandomState(seed * 1000 + j); x, y = make_cold_start_stream(X18, y18, N, 0.01, rng); b = x[:500]
        V = {"src": x, "mean_std500": (x - b.mean(0)) / (b.std(0) + 1e-6), "med_mad500": (x - np.median(b, 0)) / mad(b),
             "mean_stdall": (x - x.mean(0)) / (x.std(0) + 1e-6)}
        for k, xx in V.items():
            s = scores_of(m, xx.astype(np.float32), space); rows.append(dict(seed=seed, j=j, var=k, auroc=float(roc_auc_score(y, s)), rprec=float(rp(s, y))))
    print(pool, "seed", seed, flush=True)
json.dump(rows, open(f"round11_runA_{pool}.json", "w"))
def tci(v):
    v = np.array(v); m = v.mean(); h = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)); return m, m - h, m + h
ps = lambda k, mt: np.array([np.mean([r[mt] for r in rows if r["seed"] == s and r["var"] == k]) for s in seeds])
for k in ("src", "mean_std500", "med_mad500", "mean_stdall"):
    for mt in ("auroc", "rprec"):
        print(pool, k, mt, "%.4f [%.4f,%.4f]" % tci(ps(k, mt)), "" if k == "src" else "diff %.4f [%.4f,%.4f]" % tci(ps(k, mt) - ps("src", mt)))
