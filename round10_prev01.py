"""Round 10 (mock-review R4.3): Wed pair, AQT at true prevalence 0.1% (20 attacks per 20,000 flows) with q=0.99 and the
prevalence-matched q=0.999, 20 seeds x 4 streams, same construction as round9_extras.py prev.
usage: python3 round10_prev01.py <shard> <nshards> -> round10_prev01_shard<k>.json"""
import json, sys, numpy as np, torch
from verified_pipeline import pretrain_source_model, make_cold_start_stream
from flex_trial import score_buffer
from grid_runner import load_pool
from threshold_baselines import scores_of
from round6_frozen import fast_aqt, stats, N
torch.set_num_threads(1)
SEEDS20 = [42, 123, 456, 789, 2024] + list(range(1000, 1015))
shard, ns = int(sys.argv[1]), int(sys.argv[2]); out = {}
X17, y17, X18, y18 = load_pool("raw77"); dim = X17.shape[1]
for si, seed in enumerate(SEEDS20):
    if si % ns != shard: continue
    torch.manual_seed(seed); m = pretrain_source_model(X17, y17, dim); pre = score_buffer(m, X17, "logit")
    for prev in (0.001,):
        for j in range(4):
            rng = np.random.RandomState(seed * 1000 + j)
            Xs, ys = make_cold_start_stream(X18, y18, N, prev, rng); s = scores_of(m, Xs, "logit")
            out[f"{seed}_{j}_{prev}"] = {"q0.99": stats(fast_aqt(s, pre, 500, 0.99, 0.0), ys),
                                         "q_matched": stats(fast_aqt(s, pre, 500, 1 - prev, 0.0), ys), "n_attack": int(ys.sum())}
    print("  prev seed", seed, "done", flush=True); json.dump(out, open(f"round10_prev01_shard{shard}.json", "w"))
