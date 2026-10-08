"""Round 9/10 extras on the Wed pair (raw77, logit space, frozen model).
usage: python3 round9_extras.py prev <shard> <nshards>   -> round9_prev_shard<k>.json
         AQT attack-F1/recall vs prevalence {0.5,1,2,5,20}% at q=0.99 and prevalence-matched q, 20 seeds x 4 streams
       python3 round9_extras.py recover                  -> round9_recover.json
         decoy recovery: surrogate decoys (1% of the stream) only in the first half; recall of AQT / top-1% budget on real
         attacks before the stop and in the segments after it (3 seeds x 8 streams, as in the decoy experiment)"""
import json, sys, numpy as np, torch
from verified_pipeline import pretrain_source_model, make_cold_start_stream
from flex_trial import score_buffer
from grid_runner import load_pool
from threshold_baselines import scores_of, aqt_preds, budget_preds
from round6_frozen import fast_aqt, stats, N
torch.set_num_threads(1)
SEEDS20 = [42, 123, 456, 789, 2024] + list(range(1000, 1015))
mode = sys.argv[1]
X17, y17, X18, y18 = load_pool("raw77"); dim = X17.shape[1]
if mode == "prev":
    shard, ns = int(sys.argv[2]), int(sys.argv[3]); out = {}
    for si, seed in enumerate(SEEDS20):
        if si % ns != shard: continue
        torch.manual_seed(seed); m = pretrain_source_model(X17, y17, dim)
        pre = score_buffer(m, X17, "logit")
        for prev in (0.005, 0.01, 0.02, 0.05, 0.20):
            for j in range(4):
                rng = np.random.RandomState(seed * 1000 + j)
                Xs, ys = make_cold_start_stream(X18, y18, N, prev, rng)
                s = scores_of(m, Xs, "logit")
                out[f"{seed}_{j}_{prev}"] = {"q0.99": stats(fast_aqt(s, pre, 500, 0.99, 0.0), ys),
                                             "q_matched": stats(fast_aqt(s, pre, 500, 1 - prev, 0.0), ys)}
        print("  prev seed", seed, "done", flush=True)
        json.dump(out, open(f"round9_prev_shard{shard}.json", "w"))
else:
    out = {}
    segs = {"during": (0, 10000), "0-1000": (10000, 11000), "1000-2000": (11000, 12000), "2000+": (12000, 20000)}
    for seed in (42, 123, 456):
        torch.manual_seed(seed); m = pretrain_source_model(X17, y17, dim)
        torch.manual_seed(seed + 1000); sur = pretrain_source_model(X17, y17, dim)
        pre = score_buffer(m, X17, "logit"); s18 = scores_of(m, X18, "logit"); t18 = scores_of(sur, X18, "logit")
        ben = np.where(y18 == 0)[0]; att = np.where(y18 == 1)[0]
        top_s = ben[t18[ben] >= np.quantile(t18[ben], 0.99)]
        for j in range(8):
            for tag, ndec_first in (("decoy", 100), ("none", 0)):
                rng = np.random.RandomState(seed * 1000 + j)
                half = N // 2; n_att_h = int(round(half * 0.01))
                parts = []
                for h in (0, 1):
                    nd = ndec_first if h == 0 else 0
                    idx = np.concatenate([rng.choice(ben, half - n_att_h - nd), rng.choice(att, n_att_h),
                                          rng.choice(top_s, nd) if nd else np.array([], dtype=int)])
                    rng.shuffle(idx); parts.append(idx)
                idx = np.concatenate(parts); s = s18[idx]; y = y18[idx]
                for rule, p in (("aqt", aqt_preds(s, pre, 500, 0.99, 0.0)), ("budget", budget_preds(s, 0.01, 0.0))):
                    for sn, (a, b) in segs.items():
                        mk = y[a:b] == 1
                        out.setdefault(f"{seed}_{j}_{tag}_{rule}_{sn}", float(p[a:b][mk].mean()) if mk.any() else None)
        print("  recover seed", seed, "done", flush=True)
    json.dump(out, open("round9_recover.json", "w"))
