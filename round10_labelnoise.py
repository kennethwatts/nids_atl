"""Round 10 (mock-review R3.4): true-label fine-tuning with corrupted labels. Persistent Adam at 3x, 10 evaluation seeds
(pairs with round9_sens.py 'base' and round7 cells). Corruption of the TRAINING labels only; metrics use the true labels.
  sym<r>     every label flipped independently with probability r (r = 1%, 5%, 10% of flows)
  a2b<r>     each ATTACK label flipped to benign with probability r (r = 25%, 50%): an attacker or analyst missing attacks
usage: python3 round10_labelnoise.py <pool> <shard> <nshards> -> round10_labelnoise_<pool>_shard<k>.json"""
import json, os, sys, time
import numpy as np, torch
from verified_pipeline import pretrain_source_model, make_cold_start_stream
from flex_trial import run_flex_trial, metrics_at, score_buffer
from grid_runner import load_pool
torch.set_num_threads(1)
SEEDS = ([42, 123, 456, 789, 2024] + list(range(1000, 1015)))[:10]
pool, shard, nsh = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ck = f"round10_labelnoise_{pool}_shard{shard}.json"; done = json.load(open(ck)) if os.path.exists(ck) else {}
X17, y17, X18, y18 = load_pool(pool); dim = X17.shape[1]; t0 = time.time()
def corrupt(y, kind, r, rng):
    y2 = y.copy()
    if kind == "sym": f = rng.rand(len(y)) < r; y2[f] = 1 - y2[f]
    else: f = (y == 1) & (rng.rand(len(y)) < r); y2[f] = 0
    return y2
for u, seed in enumerate(SEEDS):
    if u % nsh != shard or str(seed) in done: continue
    torch.manual_seed(seed); m = pretrain_source_model(X17, y17, dim); state = m.state_dict(); buf = score_buffer(m, X17, "logit")
    rng = np.random.RandomState(seed * 1000); Xs, ys = make_cold_start_stream(X18, y18, 20000, 0.01, rng)
    rec = {}
    for name, (kind, r) in {"clean": ("sym", 0.0), "sym0.01": ("sym", 0.01), "sym0.05": ("sym", 0.05), "sym0.10": ("sym", 0.10),
                            "a2b0.25": ("a2b", 0.25), "a2b0.50": ("a2b", 0.50)}.items():
        yn = corrupt(ys, kind, r, np.random.RandomState(seed + 77))
        res = run_flex_trial(state, dim, Xs, yn, 20000, space="logit", aqt=True, preseed=buf, adapt="oracle", opt="persist_adam", lr_mult=3)
        res["y"] = ys.astype(np.float32)          # evaluate against the true labels
        rec[name] = metrics_at(res)["20000"]
    done[str(seed)] = rec; json.dump(done, open(ck, "w")); print(pool, shard, seed, "done", int(time.time() - t0), "s", flush=True)
