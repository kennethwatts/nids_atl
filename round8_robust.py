"""Round 8 R3-2/R3-5/R1-4/R1-5: robust clean-window thresholds and window drift (frozen model).
usage: python3 round8_robust.py <multi77|raw77>  -> round8_robust_<pool>_raw.json
 robust: window = first k benign flows with a fraction c in {0,.01,.05} replaced by attack flows.
   estimators: q99 (the Round 7 rule), q999, qadj1 = quantile 0.99*(1-0.01), qadj5 = quantile 0.99*(1-0.05)
   (the quantile of a contaminated mixture that still corresponds to the benign 99th percentile when
   the contamination is at most 1% / 5%), and qadj5_chk: qadj5 plus a contamination check (the window
   is declared contaminated when its q99 exceeds its q90 by more than 3x the benign-source q99-q90 gap,
   in which case it falls back to qadj5; otherwise q99).
 drift: threshold = q99 of benign block j (500 flows), applied to benign block j+m: benign alert rate vs m.
3 seeds x 6 draws, replay in time order (Wed 21 Feb), thinned to 1% attacks, as in round7_frozen."""
import json, sys
import numpy as np, torch
from verified_pipeline import pretrain_source_model
from threshold_baselines import scores_of
from round7_frozen import days_of, SEEDS
from round6_frozen import stats
torch.set_num_threads(1)
KS = [500, 1000, 2000, 5000]; CS = [0.0, 0.01, 0.05]

def est(win, gap_src):
    q = lambda p: float(np.quantile(win, p))
    contaminated = (q(0.99) - q(0.90)) > 3 * gap_src
    return {"q99": q(0.99), "q999": q(0.999), "qadj1": q(0.99 * 0.99), "qadj5": q(0.99 * 0.95),
            "qadj5_chk": q(0.99 * 0.95) if contaminated else q(0.99)}

pool = sys.argv[1]
X17, y17, days = days_of(pool)
out, drift = {}, {}
for seed in SEEDS:
    torch.manual_seed(seed); m = pretrain_source_model(X17, y17, X17.shape[1])
    s17 = scores_of(m, X17, "logit"); b17 = s17[y17 == 0]
    gap_src = float(np.quantile(b17, 0.99) - np.quantile(b17, 0.90))
    for d, X, y, fam in days:
        s_all = scores_of(m, X, "logit")
        benign = np.where(y == 0)[0]; attack = np.where(y == 1)[0]; n_att = int(round(0.01 / 0.99 * len(benign)))
        for k in range(6):
            rng = np.random.RandomState(seed * 100 + k)
            keep = np.sort(np.concatenate([benign, rng.choice(attack, n_att, replace=len(attack) < n_att)]))
            s, yy = s_all[keep], y[keep]; b_sc = s[yy == 0]; a_pool = s_all[attack]
            rec = {}
            for kk in KS:
                for c in CS:
                    nc = int(round(c * kk))
                    win = np.concatenate([b_sc[:kk - nc], rng.choice(a_pool, nc, replace=True)]) if nc else b_sc[:kk]
                    for name, tau in est(win, gap_src).items():
                        rec[f"{name}_k{kk}_c{c}"] = stats((s > tau).astype(np.int8), yy)
            out.setdefault(d, {}).setdefault(str(seed), []).append(rec)
        # drift: block j threshold applied to block j+m (benign scores of the day-ordered stream, seed-independent split)
        bs = s_all[y == 0]; nb = len(bs) // 500
        for mm in (0, 1, 2, 5, 10, 20):
            rates = [float((bs[(j + mm) * 500:(j + mm + 1) * 500] > np.quantile(bs[j * 500:(j + 1) * 500], 0.99)).mean())
                     for j in range(0, nb - mm)]
            drift.setdefault(d, {}).setdefault(str(mm), {})[str(seed)] = [float(np.mean(rates)), float(np.median(rates)), len(rates)]
    print(f"  robust {pool} seed {seed} done", flush=True)
json.dump(dict(robust=out, drift=drift), open(f"round8_robust_{pool}_raw.json", "w"))
