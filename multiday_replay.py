"""
Time-ordered replay on the multi-day pair, one stream per 2018 day (Round 6
Reviewer 3: replay was one day). Frozen 77-feature model pretrained on the
multi-day 2017 source; logit scores. Per day: all benign rows kept, attacks
thinned to 1% of the stream, same rows shuffled as an i.i.d. control; rules as
in round6_frozen.rules_for (AQT windows, top-1% budget, static threshold from
the first k known-clean target flows, AQT capped at that threshold).
NOTE: the multi-day pool holds a random 33,334-row draw per day (ordering and
clustering preserved, density thinned); Tue 20 Feb is itself a 1-in-26 sample.
usage: python3 multiday_replay.py
"""
import json
import numpy as np
import torch
from verified_pipeline import pretrain_source_model
from flex_trial import score_buffer
from multiday_pool import get_pool
from threshold_baselines import scores_of
from round6_frozen import rules_for, stats

torch.set_num_threads(1)
p = get_pool("77")
X17, y17, X18, y18, ts, day = p["X17"], p["y17"], p["X18"], p["y18"], p["ts18"], p["day18"]
out = {}
for seed in (42, 123, 456):
    torch.manual_seed(seed); m = pretrain_source_model(X17, y17, X17.shape[1])
    s_day = scores_of(m, X18, "logit")
    pre_by_w = {W: score_buffer(m, X17, "logit", window=W) for W in (500, 2000, 5000)}
    for d in np.unique(day):
        ix = np.where(day == d)[0]; ix = ix[np.argsort(ts[ix], kind="stable")]
        s_all, yy = s_day[ix], y18[ix]
        benign = np.where(yy == 0)[0]; attack = np.where(yy == 1)[0]
        n_att = int(round(0.01 / 0.99 * len(benign)))
        for k in range(6):
            rng = np.random.RandomState(seed * 100 + k)
            keep = np.sort(np.concatenate([benign, rng.choice(attack, n_att, replace=len(attack) < n_att)]))
            for order in ("time_ordered", "shuffled"):
                idx = keep if order == "time_ordered" else rng.permutation(keep)
                s, y = s_all[idx], yy[idx]
                rs = rules_for(s, y, pre_by_w, None)
                rec = {r: stats(pp, y) for r, pp in rs.items()}
                w = max(1, int(0.05 * len(y))); cs = np.concatenate([[0], np.cumsum(y)])
                rec["_cluster"] = {"densest5": float((cs[w:] - cs[:-w]).max() / y.sum()),
                                   "peak500": float(((cs[500:] - cs[:-500]) / 500).max())}
                out.setdefault(f"{d}|{order}", {}).setdefault(str(seed), []).append(rec)
    print("seed", seed, "done", flush=True)
json.dump(out, open("multiday_replay_raw.json", "w"))
