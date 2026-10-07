"""
Round 6 frozen-model experiments (cheap: no fine-tuning).

usage: python3 round6_frozen.py decoy <raw77|paper10>
       python3 round6_frozen.py mitig            (raw77, logit)
       python3 round6_frozen.py bbse

decoy  (Round 6 Reviewer 3 major 1): who can build the decoys? Three
       attackers insert the same fraction of benign-labelled decoys:
         random     uniformly random benign rows (needs no knowledge; a
                    control, it should do nothing)
         surrogate  rows in the top 1% of benign scores under a SURROGATE model
                    pretrained on the same source with a different seed (no
                    access to the victim's weights, thresholds or scores)
         victim     top 1% under the victim's own scores (the Round 5 attacker,
                    the strongest case: needs query or white-box access)
       Reported: recall on real attacks (1% of stream) for aqt and budget_1pct,
       and the share of surrogate-chosen decoys that really land in the
       victim's top 1% of benign scores.
mitig  (R3 major 5, R1 major 3 and 2): candidate mitigations on the real
       time-ordered replay of the raw 2018 Wednesday file (attacks thinned to
       1%), plus its shuffled control, plus each 2018 attack family alone.
       Rules: aqt (W=500), aqt with longer memory (W=2000, 5000, 20000),
       budget_1pct, static_benign_k (threshold = q99 of the first k BENIGN
       target flows, never updated: needs only a window of known-clean
       traffic), aqt_cap_benign_k (AQT, but never above that calibrated
       threshold, so a burst cannot push the threshold up).
       Also on i.i.d. streams: q sweep and prevalence-matched q (R1 major 4).
bbse   (R1 major 4): BBSE-style label-free prevalence estimate from the
       source confusion matrix, tested on the 77-feature and 10-feature pools.
"""

import json
import sys

import numpy as np
import torch

from verified_pipeline import pretrain_source_model, make_cold_start_stream
from flex_trial import score_buffer
from grid_runner import load_pool
from threshold_baselines import scores_of, aqt_preds, budget_preds
from adversarial_preseeded import real_attack_metrics

N = 20000



def fast_aqt(s, pre, W, q, default_tau=0.0, cap=None):
    """Same output as threshold_baselines.aqt_preds (sliding window of the last W
    values including the current one, np.quantile linear interpolation) but with a
    sorted window, so long windows are cheap."""
    import bisect
    from collections import deque
    win = deque(pre[-W:]); srt = sorted(win)
    out = np.zeros(len(s), dtype=np.int8)
    for t, v in enumerate(s):
        old = win.popleft(); srt.pop(bisect.bisect_left(srt, old))
        win.append(v); bisect.insort(srt, v)
        pos = (len(srt) - 1) * q; lo = int(pos); hi = min(lo + 1, len(srt) - 1)
        tau = srt[lo] + (pos - lo) * (srt[hi] - srt[lo])
        if cap is not None:
            tau = min(tau, cap)
        out[t] = v > tau
    return out

def stats(p, y):
    tp = float(((p == 1) & (y == 1)).sum()); fp = float(((p == 1) & (y == 0)).sum())
    fn = float(((p == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    return {"recall": rec, "precision": prec,
            "attack_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "alert_rate": float(p.mean())}


def decoy(pool):
    space = "logit" if pool == "raw77" else "prob"
    dtau = 0.0 if space == "logit" else 0.5
    X17, y17, X18, y18 = load_pool(pool)
    dim = X17.shape[1]
    rates = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]
    out = {}
    for seed in [42, 123, 456]:
        torch.manual_seed(seed); m = pretrain_source_model(X17, y17, dim)
        torch.manual_seed(seed + 1000); sur = pretrain_source_model(X17, y17, dim)
        pre = score_buffer(m, X17, space)
        s18 = scores_of(m, X18, space); t18 = scores_of(sur, X18, space)
        ben = np.where(y18 == 0)[0]; att = np.where(y18 == 1)[0]
        top_v = ben[s18[ben] >= np.quantile(s18[ben], 0.99)]
        top_s = ben[t18[ben] >= np.quantile(t18[ben], 0.99)]
        hit = float(np.isin(top_s, top_v).mean())
        pools = {"random": ben, "surrogate": top_s, "victim": top_v}
        for j in range(8):
            for d in rates:
                for name, dp in pools.items():
                    rng = np.random.RandomState(seed * 1000 + j)
                    n_att = int(round(N * 0.01)); n_dec = int(round(N * d)); n_ben = N - n_att - n_dec
                    idx = np.concatenate([rng.choice(ben, n_ben), rng.choice(att, n_att),
                                          rng.choice(dp, n_dec) if n_dec else np.array([], dtype=int)])
                    rng.shuffle(idx)
                    s = s18[idx]; y = y18[idx]
                    rec = {"aqt": real_attack_metrics(aqt_preds(s, pre, 500, 0.99, dtau), y),
                           "budget_1pct": real_attack_metrics(budget_preds(s, 0.01, dtau), y),
                           "surrogate_hit_rate": hit}
                    out[f"{seed}_{j}_{d}_{name}"] = rec
        print(f"  decoy {pool}: seed {seed} done (surrogate decoys in victim top-1%: {hit:.2f})", flush=True)
    json.dump(out, open(f"round6_decoy_{pool}_raw.json", "w"))


def static_from_benign(s, y_order_benign_first_k, k):
    return None


def rules_for(s, y, pre_by_w, cap_ref):
    out = {}
    benign_idx = np.where(y == 0)[0]
    for W, pre in pre_by_w.items():
        out[f"aqt_W{W}"] = fast_aqt(s, pre, W, 0.99, 0.0)
    out["budget_1pct"] = budget_preds(s, 0.01, 0.0)
    for k in (50, 500, 5000):
        tau = float(np.quantile(s[benign_idx[:k]], 0.99))
        out[f"static_benign_{k}"] = (s > tau).astype(np.int8)
        if k == 500:
            out[f"aqt_cap_benign_{k}"] = fast_aqt(s, pre_by_w[500], 500, 0.99, 0.0, cap=tau)
    return out


def mitig():
    p = load_pool_raw77()
    X17, y17, X18, y18, fam18, ts = p["X17"], p["y17"], p["X18"], p["y18"], p["fam18"], p["ts18"]
    order = np.argsort(ts, kind="stable")
    X18, y18, fam18 = X18[order], y18[order], fam18[order]
    benign = np.where(y18 == 0)[0]
    out = {}
    fams = {"all": np.where(y18 == 1)[0]}
    for f in np.unique(fam18[y18 == 1]):
        fams[f] = np.where((y18 == 1) & (fam18 == f))[0]
    print("attack families:", {k: len(v) for k, v in fams.items()}, flush=True)
    for seed in [42, 123, 456]:
        torch.manual_seed(seed); m = pretrain_source_model(X17, y17, X17.shape[1])
        s_all = scores_of(m, X18, "logit")
        pre_by_w = {W: score_buffer(m, X17, "logit", window=W) for W in (500, 2000, 5000, 20000)}
        for famname, att in fams.items():
            n_att = int(round(0.01 / 0.99 * len(benign)))
            for d in range(4):
                rng = np.random.RandomState(seed * 100 + d)
                keep = np.sort(np.concatenate([benign, rng.choice(att, min(n_att, len(att)), replace=len(att) < n_att)]))
                for order_name in ("time_ordered", "shuffled"):
                    idx = keep if order_name == "time_ordered" else rng.permutation(keep)
                    s, y = s_all[idx], y18[idx]
                    rs = rules_for(s, y, pre_by_w, None)
                    out.setdefault(f"{famname}|{order_name}", {}).setdefault(str(seed), []).append(
                        {r: stats(pp, y) for r, pp in rs.items()})
        print(f"  mitig: seed {seed} done", flush=True)
    json.dump(out, open("round6_mitigations_replay_raw.json", "w"))
    # i.i.d. q sweep and prevalence-matched q at raw77 / logit
    X17, y17, X18, y18 = load_pool("raw77")
    sw = {}
    for seed in [42, 123, 456, 789, 2024]:
        torch.manual_seed(seed); m = pretrain_source_model(X17, y17, X17.shape[1])
        pre = score_buffer(m, X17, "logit")
        for prev in (0.005, 0.01, 0.02, 0.05, 0.20):
            for j in range(4):
                rng = np.random.RandomState(seed * 1000 + j)
                Xs, ys = make_cold_start_stream(X18, y18, N, prev, rng)
                s = scores_of(m, Xs, "logit")
                rec = {f"q{q}": stats(fast_aqt(s, pre, 500, q, 0.0), ys) for q in (0.90, 0.95, 0.99, 0.995, 0.999)}
                rec["q_matched"] = stats(fast_aqt(s, pre, 500, 1 - prev, 0.0), ys)
                sw[f"{seed}_{j}_{prev}"] = rec
        print(f"  q sweep: seed {seed} done", flush=True)
    json.dump(sw, open("round6_qsweep_raw77_raw.json", "w"))


def load_pool_raw77():
    from raw_pool import get_pool
    return get_pool("77")


def bbse():
    out = {}
    for pool in ("raw77", "paper10"):
        space = "logit" if pool == "raw77" else "prob"
        X17, y17, X18, y18 = load_pool(pool)
        for seed in [42, 123, 456, 789, 2024]:
            torch.manual_seed(seed); m = pretrain_source_model(X17, y17, X17.shape[1])
            s17 = scores_of(m, X17, space); s18 = scores_of(m, X18, space)
            true_pi = float(y18.mean())
            ben = s17[y17 == 0]; atk = s17[y17 == 1]
            for name, tau in (("src_q99", float(np.quantile(ben, 0.99))), ("src_q90", float(np.quantile(ben, 0.90))),
                              ("default", 0.0 if space == "logit" else 0.5)):
                fpr = float((ben > tau).mean()); tpr = float((atk > tau).mean())
                ppr = float((s18 > tau).mean())
                est = (ppr - fpr) / (tpr - fpr) if tpr - fpr > 1e-6 else float("nan")
                out[f"{pool}|{seed}|{name}"] = dict(true_pi=true_pi, est=est, ppr=ppr, fpr=fpr, tpr=tpr)
        print(f"  bbse {pool} done", flush=True)
    json.dump(out, open("round6_bbse_raw.json", "w"))


if __name__ == "__main__":
    if sys.argv[1] == "decoy":
        decoy(sys.argv[2])
    elif sys.argv[1] == "mitig":
        mitig()
    else:
        bbse()
