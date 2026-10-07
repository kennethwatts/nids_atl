"""
Round 7 Reviewer 3 major 4: true-label fine-tuning (the paper's main positive result)
under time-ordered bursts. Per 2018 day of the multi-day pool and the Wednesday pair:
all benign rows in timestamp order, attacks thinned to 1%, plus the same rows shuffled.
Variants: aqt_alone, oracle_reset, oracle_adam_1x, oracle_adam_10x (logit space, AQT
pre-seeded, as in the grids). 3 seeds x 1 thinning draw.
usage: python3 round7_replay_ft.py <multi77|raw77> <seed>
"""
import json, sys
import numpy as np
import torch
from verified_pipeline import pretrain_source_model
from flex_trial import run_flex_trial, score_buffer
from round7_frozen import days_of
from round6_frozen import stats

torch.set_num_threads(1)
V = {"aqt_alone": dict(adapt="none"),
     "oracle_reset": dict(adapt="oracle", opt="reset_adam"),
     "oracle_adam_1x": dict(adapt="oracle", opt="persist_adam", lr_mult=1),
     "oracle_adam_10x": dict(adapt="oracle", opt="persist_adam", lr_mult=10)}

pool, seed = sys.argv[1], int(sys.argv[2])
X17, y17, days = days_of(pool)
torch.manual_seed(seed); m = pretrain_source_model(X17, y17, X17.shape[1])
state = m.state_dict(); pre = score_buffer(m, X17, "logit")
out = {}
for d, X, y, fam in days:
    benign = np.where(y == 0)[0]; attack = np.where(y == 1)[0]
    n_att = int(round(0.01 / 0.99 * len(benign)))
    rng = np.random.RandomState(seed * 100)
    keep = np.sort(np.concatenate([benign, rng.choice(attack, n_att, replace=len(attack) < n_att)]))
    for order, idx in (("time_ordered", keep), ("shuffled", rng.permutation(keep))):
        for v, kw in V.items():
            res = run_flex_trial(state, X17.shape[1], X[idx], y[idx], len(idx), space="logit", aqt=True, preseed=pre, **kw)
            out[f"{d}|{order}|{v}"] = stats(res["preds"], res["y"])
        print(f"  {pool} seed {seed} {d} {order} done", flush=True)
        json.dump(out, open(f"round7_replayft_{pool}_{seed}.json", "w"))
