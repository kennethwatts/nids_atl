"""Round 9 (mock-review Run B): SPOT (Siffer et al., KDD 2017) against AQT, the cumulative top-1% budget and a clean-500
static threshold, on frozen-model scores. SPOT: initial threshold t = 98th percentile of 2,000 source scores, generalized
Pareto fit to excesses, alert threshold z_q for risk q=1e-2 (and 1e-3); alerts are NOT added to the model, other exceedances of t update it.
Streams: (a) Wed pair (raw77) and multi-day (multi77) days, time-ordered and shuffled, attacks thinned to 1%
(seeds 42,123,456); (b) decoy streams of round8_surrogate.py (raw77; random and disjoint-half surrogate; rates 0,1,5%).
SPOT variant calibrated on the first 2,000 target scores. usage: python3 round9_spot_tgt.py -> round9_spot_tgt.json"""
import json, sys, warnings
import numpy as np, torch
from scipy.stats import genpareto
from verified_pipeline import pretrain_source_model
from flex_trial import score_buffer
from grid_runner import load_pool
from threshold_baselines import scores_of, aqt_preds, budget_preds
from adversarial_preseeded import real_attack_metrics
from round7_frozen import days_of
from round6_frozen import stats
warnings.filterwarnings("ignore")
torch.set_num_threads(1)

class SPOT:
    def __init__(self, init, q=1e-2, level=0.98):
        init = np.asarray(init, float); self.q = q
        self.t = float(np.quantile(init, level)); self.peaks = list(init[init > self.t] - self.t)
        self.n = len(init); self._fit()
    def _fit(self):
        # generalized Pareto fit to the excesses by probability-weighted moments (Hosking & Wallis 1987); the scipy
        # MLE was unstable on 40 to 400 peaks (it returned thresholds below t in a sanity check)
        Y = np.sort(np.asarray(self.peaks)); Nt = len(Y)
        if Nt < 5:
            self.zq = self.t + (Y.mean() if Nt else 0.0) * 3; return
        p = (np.arange(1, Nt + 1) - 0.35) / Nt; a0 = Y.mean(); a1 = np.mean((1 - p) * Y)
        k = float(np.clip(a0 / (a0 - 2 * a1) - 2, -0.9, 0.9)); sg = 2 * a0 * a1 / (a0 - 2 * a1)
        r = self.q * self.n / Nt
        self.zq = self.t + (sg / k) * (1 - r ** k) if abs(k) > 1e-6 else self.t - sg * np.log(r)
        self.zq = max(self.zq, self.t)
    def run(self, s):
        p = np.zeros(len(s), np.int8)
        for i, x in enumerate(s):
            if x > self.zq: p[i] = 1
            elif x > self.t:
                self.peaks.append(x - self.t); self.n += 1; self._fit()
            else: self.n += 1
        return p

def spot_tgt(s, q, n0=2000):
    """SPOT calibrated on the first n0 target scores (label-free); those flows raise no alert."""
    p = np.zeros(len(s), np.int8); p[n0:] = SPOT(s[:n0], q=q).run(s[n0:]); return p

def rules(s, pre, init, y):
    return {"spot_tgt": spot_tgt(s, 1e-2), "spot_tgt_q1e-3": spot_tgt(s, 1e-3)}

out = {"replay": {}, "decoy": {}}
for pool in ("raw77", "multi77"):
    X17, y17, days = days_of(pool); dim = X17.shape[1]
    for seed in (42, 123, 456):
        torch.manual_seed(seed); m = pretrain_source_model(X17, y17, dim)
        pre = score_buffer(m, X17, "logit"); init = score_buffer(m, X17, "logit", window=2000)
        for d, X, y, fam in days:
            sc = scores_of(m, X, "logit"); ben = np.where(y == 0)[0]; att = np.where(y == 1)[0]
            n_att = int(round(0.01 / 0.99 * len(ben)))
            for draw in range(5):
                rng = np.random.RandomState(seed * 100 + draw)
                keep = np.sort(np.concatenate([ben, rng.choice(att, n_att, replace=len(att) < n_att)]))
                for order, idx in (("time_ordered", keep), ("shuffled", rng.permutation(keep))):
                    s, yy = sc[idx], y[idx]
                    for r, p in rules(s, pre, init, yy).items():
                        out["replay"].setdefault(f"{pool}|{d}|{order}|{r}", []).append(dict(seed=seed, draw=draw, **stats(p, yy)))
            print(pool, seed, d, "done", flush=True)
    json.dump(out, open("round9_spot_tgt.json", "w"))

X17, y17, X18, y18 = load_pool("raw77"); dim = X17.shape[1]; N = 20000
for seed in (42, 123, 456):
    rs = np.random.RandomState(seed); perm = rs.permutation(len(X17)); A, B = perm[: len(perm) // 2], perm[len(perm) // 2:]
    torch.manual_seed(seed); vic = pretrain_source_model(X17[A], y17[A], dim)
    torch.manual_seed(seed + 2000); sur = pretrain_source_model(X17[B], y17[B], dim)
    pre = score_buffer(vic, X17[A], "logit"); init = score_buffer(vic, X17[A], "logit", window=2000)
    sv = scores_of(vic, X18, "logit"); ss = scores_of(sur, X18, "logit")
    ben = np.where(y18 == 0)[0]; att = np.where(y18 == 1)[0]
    pools = {"random": ben, "disjoint_B": ben[ss[ben] >= np.quantile(ss[ben], 0.99)]}
    for name, dp in pools.items():
        for rate in (0.0, 0.01, 0.05):
            for j in range(8):
                rng = np.random.RandomState(seed * 1000 + j)
                n_att = int(round(N * 0.01)); n_dec = int(round(N * rate)); n_ben = N - n_att - n_dec
                idx = np.concatenate([rng.choice(ben, n_ben), rng.choice(att, n_att), rng.choice(dp, n_dec) if n_dec else np.array([], dtype=int)])
                rng.shuffle(idx); s, yy = sv[idx], y18[idx]
                for r, p in rules(s, pre, init, yy).items():
                    pass
                    out["decoy"].setdefault(f"{name}|{rate}|{r}", []).append(dict(seed=seed, **real_attack_metrics(p, yy)))
    print("decoy seed", seed, "done", flush=True)
    json.dump(out, open("round9_spot_tgt.json", "w"))
