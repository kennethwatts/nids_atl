"""Base-model ranking check on the multi-day pair (logit space), per target family and per day."""
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from verified_pipeline import pretrain_source_model
from flex_trial import model_logit
from multiday_pool import get_pool
torch.set_num_threads(1)
for fs in ("77", "10"):
    p = get_pool(fs); out = []
    for seed in (42, 123, 456):
        torch.manual_seed(seed)
        m = pretrain_source_model(p["X17"], p["y17"], p["X17"].shape[1]); m.eval()
        with torch.no_grad():
            s = model_logit(m, torch.tensor(p["X18"])).numpy().ravel()
        res = {"all": roc_auc_score(p["y18"], s)}
        for fam in np.unique(p["fam18"]):
            if fam.upper() == "BENIGN": continue
            mk = (p["fam18"] == fam) | (p["y18"] == 0)
            if (p["fam18"] == fam).sum() >= 50:
                res[fam] = roc_auc_score(p["y18"][mk], s[mk])
        out.append(res)
    print(fs, {k: round(float(np.mean([r[k] for r in out])), 3) for k in out[0]}, flush=True)
