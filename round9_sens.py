"""Round 9: sensitivity of true-label fine-tuning to the sigmoid schedule (k, x0).
Persistent Adam at 3x (the held-out rate in raw77/multi77/nf), 10 of the 20 evaluation seeds, same pretraining
seed and stream as round7_grid.py 'cells' (seed*1000+0), so each cell pairs with round7_cells aqt/oracle_adam_3x.
usage: python3 round9_sens.py <pool> <shard> <nshards>  -> round9_sens_<pool>_shard<k>.json"""
import json, os, sys, time
import numpy as np, torch
import flex_trial
from verified_pipeline import pretrain_source_model, make_cold_start_stream
from flex_trial import run_flex_trial, metrics_at, score_buffer
from grid_runner import load_pool
torch.set_num_threads(1)
SEEDS = ([42, 123, 456, 789, 2024] + list(range(1000, 1015)))[:10]
CFG = {"base": (0.001, 3600.0), "k0.0005": (0.0005, 3600.0), "k0.002": (0.002, 3600.0),
       "x0_1800": (0.001, 1800.0), "x0_7200": (0.001, 7200.0)}
pool, shard, nsh = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ck = f"round9_sens_{pool}_shard{shard}.json"
done = json.load(open(ck)) if os.path.exists(ck) else {}
X17, y17, X18, y18 = load_pool(pool); dim = X17.shape[1]; t0 = time.time()
for u, seed in enumerate(SEEDS):
    if u % nsh != shard or str(seed) in done: continue
    torch.manual_seed(seed); m = pretrain_source_model(X17, y17, dim); state = m.state_dict()
    buf = score_buffer(m, X17, "logit")
    rng = np.random.RandomState(seed * 1000); Xs, ys = make_cold_start_stream(X18, y18, 20000, 0.01, rng)
    rec = {}
    for name, (k, x0) in CFG.items():
        flex_trial.sigmoid_weight = (lambda t, k=k, x0=x0: 1.0 / (1.0 + np.exp(-k * (t - x0))))
        res = run_flex_trial(state, dim, Xs, ys, 20000, space="logit", aqt=True, preseed=buf, adapt="oracle", opt="persist_adam", lr_mult=3)
        rec[name] = metrics_at(res)["20000"]
    flex_trial.sigmoid_weight = (lambda t: 1.0 / (1.0 + np.exp(-0.001 * (t - 3600.0))))
    res = run_flex_trial(state, dim, Xs, ys, 20000, space="logit", aqt=True, preseed=buf, adapt="none")
    rec["aqt"] = metrics_at(res)["20000"]
    done[str(seed)] = rec; json.dump(done, open(ck, "w"))
    print(pool, shard, seed, "done", int(time.time() - t0), "s", flush=True)
