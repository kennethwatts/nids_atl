"""Round 8 R3-1: does true-label fine-tuning rescue the burst when the alert rule is not AQT?
Time-ordered and shuffled replay (as round7_replay_ft.py) of fine-tuned models; the SAME online scores are
then thresholded by: aqt (W=500), budget_1pct (causal top 1%), clean500 (static q99 of the first 500
benign flows' scores, a known-clean window), and the non-causal ceiling (q99 of all benign scores).
Variants: frozen (no updates), oracle_adam_1x, oracle_adam_10x.
usage: python3 round8_replay_thr.py <multi77|raw77> <seed> -> round8_replaythr_<pool>_<seed>.json"""
import json, sys
import numpy as np, torch
from verified_pipeline import pretrain_source_model
from flex_trial import run_flex_trial, score_buffer
from threshold_baselines import budget_preds
from round7_frozen import days_of
from round6_frozen import stats
torch.set_num_threads(1)
V = {"frozen": dict(adapt="none"),
     "oracle_adam_1x": dict(adapt="oracle", opt="persist_adam", lr_mult=1),
     "oracle_adam_10x": dict(adapt="oracle", opt="persist_adam", lr_mult=10)}
pool, seed = sys.argv[1], int(sys.argv[2])
X17, y17, days = days_of(pool)
torch.manual_seed(seed); m = pretrain_source_model(X17, y17, X17.shape[1])
state = m.state_dict(); pre = score_buffer(m, X17, "logit")
out = {}
for d, X, y, fam in days:
    benign = np.where(y == 0)[0]; attack = np.where(y == 1)[0]; n_att = int(round(0.01 / 0.99 * len(benign)))
    rng = np.random.RandomState(seed * 100)
    keep = np.sort(np.concatenate([benign, rng.choice(attack, n_att, replace=len(attack) < n_att)]))
    for order, idx in (("time_ordered", keep), ("shuffled", rng.permutation(keep))):
        for v, kw in V.items():
            res = run_flex_trial(state, X17.shape[1], X[idx], y[idx], len(idx), space="logit", aqt=True, preseed=pre, **kw)
            s, yy = res["scores"], res["y"]; b = s[yy == 0]
            rules = {"aqt": res["preds"], "budget_1pct": budget_preds(s, 0.01, 0.0),
                     "clean500": (s > np.quantile(b[:500], 0.99)).astype(np.int8),
                     "ceiling": (s > np.quantile(b, 0.99)).astype(np.int8)}
            for r, p in rules.items():
                out[f"{d}|{order}|{v}|{r}"] = stats(p, yy)
        print(f"  {pool} seed {seed} {d} {order} done", flush=True)
        json.dump(out, open(f"round8_replaythr_{pool}_{seed}.json", "w"))
