"""TPR at 1% FPR and AUROC of the frozen model on each pool's full target set (3 pretraining seeds),
the ranking variable a top-1% budget rule actually depends on (Round 7 Reviewer 1 major 3)."""
import json, numpy as np, torch
from sklearn.metrics import roc_auc_score
from verified_pipeline import pretrain_source_model
from grid_runner import load_pool
from threshold_baselines import scores_of
torch.set_num_threads(1)
out = {}
for pool, space in (("paper10", "prob"), ("raw77", "logit"), ("multi77", "logit"), ("multi10", "logit")):
    X17, y17, X18, y18 = load_pool(pool); r = []
    for seed in (42, 123, 456):
        torch.manual_seed(seed); m = pretrain_source_model(X17, y17, X17.shape[1])
        s = scores_of(m, X18, "logit" if space == "logit" else "prob")
        thr = np.quantile(s[y18 == 0], 0.99)
        r.append(dict(auroc=float(roc_auc_score(y18, s)), tpr1=float((s[y18 == 1] > thr).mean())))
    out[pool] = {k: float(np.mean([x[k] for x in r])) for k in r[0]}
    print(pool, out[pool], flush=True)
json.dump(out, open("round7_tpr1.json", "w"))
