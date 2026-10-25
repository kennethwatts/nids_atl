"""Round 10 (mock-review R3.3): decoys timed just before the burst. Time-ordered replay (attacks thinned to 1%, as
round9_spot.py) of Wed 21 Feb (raw77) and the multi-day days Tue 20 and Wed 21 (multi77). The victim is trained on half A of
the source, a surrogate on the disjoint half B (as round8_surrogate.py); D decoys (benign flows in the surrogate's top 1% of benign
scores) are inserted at random positions in the 1,000 flows before the start of the densest 5% attack window.
Rules on the victim's frozen scores: AQT (W=500), cumulative top-1% budget, clean-500 static threshold, SPOT calibrated on the first
2,000 target scores (risk 1e-2). Metric: recall on real attacks. 3 seeds x 4 draws. -> round10_timed_decoy.json"""
import json, warnings, numpy as np, torch
exec(open("round9_spot.py").read().split("out = {")[0])
from round7_frozen import days_of
def spot_tgt(s, q, n0=2000):
    """SPOT calibrated on the first n0 target scores (label-free); those flows raise no alert."""
    p = np.zeros(len(s), np.int8); p[n0:] = SPOT(s[:n0], q=q).run(s[n0:]); return p


out = {}
for pool in ("raw77", "multi77"):
    X17, y17, days = days_of(pool); dim = X17.shape[1]
    for seed in (42, 123, 456):
        rs = np.random.RandomState(seed); perm = rs.permutation(len(X17)); A, B = perm[: len(perm) // 2], perm[len(perm) // 2:]
        torch.manual_seed(seed); vic = pretrain_source_model(X17[A], y17[A], dim)
        torch.manual_seed(seed + 2000); sur = pretrain_source_model(X17[B], y17[B], dim)
        pre = score_buffer(vic, X17[A], "logit")
        for d, X, y, fam in days:
            if pool == "multi77" and d == "wed14_18": continue
            sv = scores_of(vic, X, "logit"); ss = scores_of(sur, X, "logit")
            ben = np.where(y == 0)[0]; att = np.where(y == 1)[0]; dp = ben[ss[ben] >= np.quantile(ss[ben], 0.99)]
            n_att = int(round(0.01 / 0.99 * len(ben)))
            for draw in range(4):
                rng = np.random.RandomState(seed * 100 + draw)
                keep = np.sort(np.concatenate([ben, rng.choice(att, n_att, replace=len(att) < n_att)]))
                s0, y0 = sv[keep], y[keep]; ap = np.where(y0 == 1)[0]; L = max(1, int(0.05 * len(y0)))
                cnt = np.array([((ap >= a) & (ap < a + L)).sum() for a in range(0, len(y0) - L, max(1, L // 10))])
                start = int(np.argmax(cnt)) * max(1, L // 10); first = int(ap[ap >= start][0])
                for D in (0, 100, 500):
                    pos = np.sort(rng.randint(max(0, first - 1000), first + 1, D)) if D else np.array([], int)
                    dsc = sv[rng.choice(dp, D)] if D else np.array([])
                    s = np.insert(s0, pos, dsc); yy = np.insert(y0, pos, 0)
                    P = {"aqt": aqt_preds(s, pre, 500, 0.99, 0.0), "budget_1pct": budget_preds(s, 0.01, 0.0),
                         "clean500": (s > np.quantile(s[yy == 0][:500], 0.99)).astype(np.int8), "spot_tgt": spot_tgt(s, 1e-2)}
                    for r, p in P.items():
                        out.setdefault(f"{pool}|{d}|{D}|{r}", []).append(dict(seed=seed, draw=draw, **real_attack_metrics(p, yy)))
            print(pool, seed, d, "done", flush=True)
        json.dump(out, open("round10_timed_decoy.json", "w"))
