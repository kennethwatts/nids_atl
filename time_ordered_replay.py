"""
Time-ordered replay of the raw 2018 day file (Round 5 Reviewer 3 major
concern 4: "The stream is still i.i.d. Bursts are synthetic. At minimum,
replay the raw day files in time order to test real drift and clustering.")

The cold-start stream used everywhere else resamples the target pool i.i.d.
with replacement. Here the cleaned 2018 target pool (raw_pool.py, 77 features,
logit-space scoring) is instead replayed in timestamp order. To keep the 1%
prevalence the AQT quantile assumes, all benign rows are kept and the attack
rows are randomly thinned to 1% of the final stream; the attacks keep their
real timestamps, so their natural clustering (DDoS happens in bursts) is
preserved. The identical selected rows are also replayed in shuffled order as
an i.i.d. control. Frozen model, 3 pretraining seeds x 10 thinning draws.

Reported per rule (aqt, aqt_cap, src_q99, budget_1pct, frozen_static):
overall recall, precision, alert rate, attack-F1; and clustering diagnostics:
the share of attacks inside the densest 5%-of-stream window and the peak local
attack prevalence over a 500-sample window.

NOTE: timestamps in the cleaned pool are the capture timestamps of the
retained rows; only ordering is used, never absolute time.
"""

import json

import numpy as np
import torch

from verified_pipeline import pretrain_source_model
from flex_trial import score_buffer
from raw_pool import get_pool
from threshold_baselines import scores_of, aqt_preds, budget_preds

SEEDS = [42, 123, 456]
DRAWS = 10


def stats(preds, y):
    tp = float(((preds == 1) & (y == 1)).sum()); fp = float(((preds == 1) & (y == 0)).sum())
    fn = float(((preds == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    return {"recall": rec, "precision": prec,
            "attack_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "alert_rate": float(preds.mean())}


def main():
    p = get_pool("77")
    X17, y17, X18, y18, ts = p["X17"], p["y17"], p["X18"], p["y18"], p["ts18"]
    order = np.argsort(ts, kind="stable")
    X18, y18 = X18[order], y18[order]
    benign = np.where(y18 == 0)[0]; attack = np.where(y18 == 1)[0]
    n_att = int(round(0.01 / 0.99 * len(benign)))
    print(f"benign rows {len(benign)}, attack rows {len(attack)}, thinned attacks kept {n_att}")
    out = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        m = pretrain_source_model(X17, y17, X17.shape[1])
        s17 = scores_of(m, X17, "logit")
        cap = float(np.quantile(s17[y17 == 0], 0.99))
        pre = score_buffer(m, X17, "logit")
        s_all = scores_of(m, X18, "logit")
        for d in range(DRAWS):
            rng = np.random.RandomState(seed * 100 + d)
            keep = np.sort(np.concatenate([benign, rng.choice(attack, n_att, replace=False)]))
            for order_name in ("time_ordered", "shuffled"):
                idx = keep if order_name == "time_ordered" else rng.permutation(keep)
                s, y = s_all[idx], y18[idx]
                preds = {"aqt": aqt_preds(s, pre, 500, 0.99, 0.0),
                         "aqt_cap": aqt_preds(s, pre, 500, 0.99, 0.0, cap=cap),
                         "src_q99": (s > cap).astype(np.int8),
                         "budget_1pct": budget_preds(s, 0.01, 0.0),
                         "frozen_static": (s > 0.0).astype(np.int8)}
                rec = {r: stats(pp, y) for r, pp in preds.items()}
                # clustering diagnostics
                w = max(1, int(0.05 * len(y)))
                cs = np.concatenate([[0], np.cumsum(y)])
                dens = (cs[w:] - cs[:-w])
                rec["_cluster"] = {"share_attacks_in_densest_5pct": float(dens.max() / y.sum()),
                                   "peak_local_prevalence_500": float(((cs[500:] - cs[:-500]) / 500).max())}
                out.setdefault(order_name, {}).setdefault(str(seed), []).append(rec)
        print(f"  seed {seed} done", flush=True)
    json.dump(out, open("time_ordered_replay_raw.json", "w"))
    print("saved time_ordered_replay_raw.json")


if __name__ == "__main__":
    main()
