"""
Round 7 frozen-model studies (cheap: scoring only).

cleanwin  Reviewer 1 major 1, Reviewer 3 majors 2-3. Time-ordered replay per 2018 day
          (multi-day pool) and for the Wednesday pair. Static threshold = 99th
          percentile of the first k clean target flows; k in {50..5000}; the window
          is contaminated with 0, 1 and 5% attack flows. Also top-1% budget, AQT,
          a non-causal static ceiling (99th percentile of all benign scores in the
          stream) and the source-benign static threshold. Also writes: ranking
          summaries (AUROC, TPR at 1% FPR, per family), the benign-score
          99th-percentile drift over blocks of 500 benign flows, and an AQT
          timeline (local prevalence, tau, alerts) for the figure.
dose      Reviewer 2 major 5. AQT versus ranking quality: Gaussian noise (in units
          of the benign score std) added to frozen logits on the Wednesday pair and
          on each multi-day 2018 day, i.i.d. 1% streams, plus the natural
          (noise-free) points.
usage: python3 round7_frozen.py <cleanwin|dose> [pool: multi77|raw77]
"""
import json, sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score

from verified_pipeline import pretrain_source_model
from flex_trial import score_buffer
from threshold_baselines import scores_of, budget_preds
from round6_frozen import fast_aqt, stats

torch.set_num_threads(1)
SEEDS = [42, 123, 456]
KS = [50, 100, 200, 500, 1000, 2000, 5000]
CS = [0.0, 0.01, 0.05]


def days_of(pool):
    """-> X17, y17, [(day name, X18 rows, y, fam, ts)] sorted by timestamp."""
    if pool == "multi77":
        from multiday_pool import get_pool
        p = get_pool("77"); out = []
        for d in np.unique(p["day18"]):
            ix = np.where(p["day18"] == d)[0]; ix = ix[np.argsort(p["ts18"][ix], kind="stable")]
            out.append((d, p["X18"][ix], p["y18"][ix], p["fam18"][ix]))
    else:
        from raw_pool import get_pool
        p = get_pool("77"); ix = np.argsort(p["ts18"], kind="stable")
        out = [("wed21_raw", p["X18"][ix], p["y18"][ix], p["fam18"][ix])]
    return p["X17"], p["y17"], out


def ranking(s, y, fam):
    b = s[y == 0]; thr = np.quantile(b, 0.99); res = {"all": {"auroc": float(roc_auc_score(y, s)), "tpr1": float((s[y == 1] > thr).mean())}}
    for f in np.unique(fam[y == 1]):
        m = fam == f
        if m.sum() < 50: continue
        mk = m | (y == 0)
        res[f] = {"auroc": float(roc_auc_score(y[mk], s[mk])), "tpr1": float((s[m] > thr).mean()), "n": int(m.sum())}
    return res


def aqt_taus(s, pre, W=500, q=0.99):
    import bisect
    from collections import deque
    win = deque(pre[-W:]); srt = sorted(win); taus = np.zeros(len(s))
    for t, v in enumerate(s):
        old = win.popleft(); srt.pop(bisect.bisect_left(srt, old)); win.append(v); bisect.insort(srt, v)
        pos = (len(srt) - 1) * q; lo = int(pos); hi = min(lo + 1, len(srt) - 1)
        taus[t] = srt[lo] + (pos - lo) * (srt[hi] - srt[lo])
    return taus


def cleanwin(pool):
    X17, y17, days = days_of(pool)
    out, rank, drift, timeline = {}, {}, {}, {}
    for seed in SEEDS:
        torch.manual_seed(seed); m = pretrain_source_model(X17, y17, X17.shape[1])
        s17 = scores_of(m, X17, "logit"); src_q99 = float(np.quantile(s17[y17 == 0], 0.99))
        pre = score_buffer(m, X17, "logit")
        for d, X, y, fam in days:
            s_all = scores_of(m, X, "logit")
            rank.setdefault(d, {})[str(seed)] = ranking(s_all, y, fam)
            benign = np.where(y == 0)[0]; attack = np.where(y == 1)[0]
            n_att = int(round(0.01 / 0.99 * len(benign)))
            for k in range(6):
                rng = np.random.RandomState(seed * 100 + k)
                keep = np.sort(np.concatenate([benign, rng.choice(attack, n_att, replace=len(attack) < n_att)]))
                s, yy = s_all[keep], y[keep]
                b_sc = s[yy == 0]; a_pool = s_all[attack]
                rec = {"static_ceiling": stats((s > np.quantile(b_sc, 0.99)).astype(np.int8), yy),
                       "src_q99": stats((s > src_q99).astype(np.int8), yy),
                       "budget_1pct": stats(budget_preds(s, 0.01, 0.0), yy),
                       "aqt_W500": stats(fast_aqt(s, pre, 500, 0.99, 0.0), yy)}
                for kk in KS:
                    for c in CS:
                        nc = int(round(c * kk))
                        win = np.concatenate([b_sc[:kk - nc], rng.choice(a_pool, nc, replace=True)]) if nc else b_sc[:kk]
                        rec[f"win_k{kk}_c{c}"] = stats((s > np.quantile(win, 0.99)).astype(np.int8), yy)
                out.setdefault(d, {}).setdefault(str(seed), []).append(rec)
                if seed == SEEDS[0] and k == 0:
                    nb = len(b_sc) // 500
                    drift[d] = [float(np.quantile(b_sc[i * 500:(i + 1) * 500], 0.99)) for i in range(nb)]
                    taus = aqt_taus(s, pre); pr = fast_aqt(s, pre, 500, 0.99, 0.0)
                    cs = np.concatenate([[0], np.cumsum(yy)]); loc = (cs[500:] - cs[:-500]) / 500
                    step = 50
                    timeline[d] = {"t": list(range(500, len(s), step)), "local_prev": [float(loc[t - 500]) for t in range(500, len(s), step)],
                                   "tau": [float(taus[t]) for t in range(500, len(s), step)],
                                   "alert_rate_500": [float(pr[t - 500:t].mean()) for t in range(500, len(s), step)],
                                   "src_q99": src_q99, "win500_tau": float(np.quantile(b_sc[:500], 0.99))}
        print(f"  cleanwin {pool} seed {seed} done", flush=True)
    json.dump(dict(replay=out, rank=rank, drift=drift, timeline=timeline), open(f"round7_cleanwin_{pool}_raw.json", "w"))


def macro(p, y):
    return float(f1_score(y, p, average="macro", zero_division=0))


def dose(pool):
    from verified_pipeline import make_cold_start_stream
    SIG = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    X17, y17, days = days_of(pool)
    out = {}
    for seed in SEEDS:
        torch.manual_seed(seed); m = pretrain_source_model(X17, y17, X17.shape[1])
        s17 = scores_of(m, X17, "logit"); pre0 = score_buffer(m, X17, "logit")
        sd17 = float(np.std(s17[y17 == 0]))
        for d, X, y, fam in days:
            s_all = scores_of(m, X, "logit")
            for sg in SIG:
                for rep in range(4):
                    rng = np.random.RandomState(seed * 1000 + rep)
                    noise = np.random.RandomState(seed * 7 + rep).randn(len(s_all)) * sg * sd17
                    sn = s_all + noise
                    # i.i.d. 1% stream from this day's rows (same construction as the grids)
                    ix = np.arange(len(sn)); Xi, yi = make_cold_start_stream(ix.reshape(-1, 1).astype(np.float32), y, 20000, 0.01, rng)
                    idx = Xi.ravel().astype(int); s, yy = sn[idx], y[idx]
                    # pre-seed with source-benign+attack scores plus the same noise level
                    pre = list((pre0 + np.random.RandomState(seed * 11 + rep).randn(len(pre0)) * sg * sd17))
                    b = s[yy == 0]; thr = np.quantile(b, 0.99)
                    rk = {"auroc": float(roc_auc_score(yy, s)), "tpr1": float((s[yy == 1] > thr).mean())}
                    p_aqt = fast_aqt(s, pre, 500, 0.99, 0.0); p_bud = budget_preds(s, 0.01, 0.0); p_base = (s > 0).astype(np.int8)
                    out.setdefault(f"{d}|{sg}", []).append(dict(seed=seed, **rk,
                        aqt_macro=macro(p_aqt, yy), base_macro=macro(p_base, yy), budget_macro=macro(p_bud, yy),
                        aqt_af1=stats(p_aqt, yy)["attack_f1"], base_af1=stats(p_base, yy)["attack_f1"], budget_af1=stats(p_bud, yy)["attack_f1"]))
        print(f"  dose {pool} seed {seed} done", flush=True)
    json.dump(out, open(f"round7_dose_{pool}_raw.json", "w"))


if __name__ == "__main__":
    {"cleanwin": cleanwin, "dose": dose}[sys.argv[1]](sys.argv[2])
