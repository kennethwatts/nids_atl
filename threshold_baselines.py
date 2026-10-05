"""
Threshold-rule baselines, prevalence mismatch, hybrid cap, window length and
attack-class metrics with CIs (Round 5 Reviewer 2 major 4, Reviewer 1 major
4, Reviewer 3 major 3).

Everything here is frozen-model scoring (no fine-tuning), so it is cheap.

Rules compared on the same paired streams (5 pretraining seeds x 6 streams):
  aqt           pre-seeded sliding-window quantile q=0.99, W=500 (the paper)
  aqt_w2000     same with W=2000 (longer window)
  aqt_cap       AQT with its threshold capped at the source-benign 99th
                percentile (hybrid: the target can lower the threshold but
                never push it above what the source domain supports)
  src_q99       STATIC threshold fixed before the stream at the 99th
                percentile of the frozen model's scores on SOURCE-BENIGN rows
  budget_1pct   CAUSAL cumulative top-1% rule: alert if the score ranks in the
                top 1% of all scores seen so far in this stream (needs no
                pre-seed or window, but hard-codes a 1% alert budget, i.e.
                it encodes the prevalence)
  oracle_topk   NON-CAUSAL upper bound: threshold at the stream's own
                (1 - prevalence) quantile, computed from the whole stream.
                Reported only as a ceiling for any alert-budget rule.
At prevalences {0.5%, 1%, 2%, 5%, 20%}. budget_1pct and aqt (q=0.99) keep the
1% budget at every prevalence, so at 0.5% they over-alert and at 5% and 20%
they are structurally capped: recall <= alert_rate / prevalence.

Metrics at n=20,000: attack-F1, recall, precision, alert rate, macro-F1, with
cluster-bootstrap 95% CIs over seeds and streams. Pools: paper10 (probability
and logit) and raw77 (logit).

usage: python3 threshold_baselines.py <paper10|raw77> <space>
"""

import bisect
import json
import sys

import numpy as np
import torch

from verified_pipeline import pretrain_source_model, make_cold_start_stream
from flex_trial import model_logit, score_buffer, metrics_at
from grid_runner import load_pool

SEEDS = [42, 123, 456, 789, 2024]
STREAMS = 6
PREVALENCES = [0.005, 0.01, 0.02, 0.05, 0.20]
N = 20000
Q = 0.99


def scores_of(model, X, space):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32)
        s = model_logit(model, x) if space == "logit" else model(x)
    return s.numpy().flatten()


def aqt_preds(s, preseed, window, q, default_tau, cap=None):
    buf = list(preseed)
    preds = np.zeros(len(s), dtype=np.int8)
    for t, v in enumerate(s):
        buf.append(v)
        tau = float(np.quantile(buf[-window:], q)) if len(buf) >= window else default_tau
        if cap is not None:
            tau = min(tau, cap)
        preds[t] = v > tau
    return preds


def budget_preds(s, frac, default_tau, min_hist=100):
    srt = []
    preds = np.zeros(len(s), dtype=np.int8)
    for t, v in enumerate(s):
        bisect.insort(srt, v)
        if len(srt) >= min_hist:
            k = int(np.ceil((1 - frac) * len(srt))) - 1
            tau = srt[min(max(k, 0), len(srt) - 1)]
        else:
            tau = default_tau
        preds[t] = v > tau
    return preds


def main(pool, space):
    X17, y17, X18, y18 = load_pool(pool)
    dim = X17.shape[1]
    default_tau = 0.0 if space == "logit" else 0.5
    out = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        m = pretrain_source_model(X17, y17, dim)
        s17 = scores_of(m, X17, space)
        src_q99 = float(np.quantile(s17[y17 == 0], 0.99))
        pre = score_buffer(m, X17, space)
        pre2000 = score_buffer(m, X17, space, window=2000)
        for prev in PREVALENCES:
            for j in range(STREAMS):
                rng = np.random.RandomState(seed * 1000 + j)
                Xs, ys = make_cold_start_stream(X18, y18, N, prev, rng)
                s = scores_of(m, Xs, space)
                preds = {
                    "aqt": aqt_preds(s, pre, 500, Q, default_tau),
                    "aqt_w2000": aqt_preds(s, pre2000, 2000, Q, default_tau),
                    "aqt_cap": aqt_preds(s, pre, 500, Q, default_tau, cap=src_q99),
                    "src_q99": (s > src_q99).astype(np.int8),
                    "budget_1pct": budget_preds(s, 0.01, default_tau),
                    "oracle_topk": (s > np.quantile(s, 1 - prev)).astype(np.int8),
                    "frozen_static": (s > default_tau).astype(np.int8),
                }
                for rule, p in preds.items():
                    res = dict(preds=p, y=ys.astype(np.float32))
                    out.setdefault(f"{seed}_{j}_{prev}", {})[rule] = metrics_at(res, [N])[str(N)]
        print(f"  seed {seed} done", flush=True)
    name = f"threshold_baselines_{pool}_{space}"
    json.dump(out, open(name + "_raw.json", "w"))
    print("saved", name + "_raw.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
