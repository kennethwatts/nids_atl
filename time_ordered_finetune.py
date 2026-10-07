"""
Time-ordered replay with the fine-tuned variants (Round 6 Reviewer 3 major 4:
"report replay with the fine-tuned variants, not just the frozen model").

Same stream construction as time_ordered_replay.py (raw 2018 Wednesday file in
timestamp order, all benign rows, attacks thinned to 1%), 77 features, logit
space, 3 pretraining seeds x 3 thinning draws, each variant on both the
time-ordered stream and a shuffled copy of the same rows.
Variants: aqt_alone, oracle_reset, pseudo_reset, pseudo_adam1x.
"""
import json
import numpy as np
import torch
from verified_pipeline import pretrain_source_model
from flex_trial import run_flex_trial, score_buffer
from raw_pool import get_pool

SEEDS = [42, 123, 456]
DRAWS = 3
V = {"aqt_alone": dict(adapt="none"),
     "oracle_reset": dict(adapt="oracle", opt="reset_adam"),
     "pseudo_reset": dict(adapt="pseudo", opt="reset_adam"),
     "pseudo_adam1x": dict(adapt="pseudo", opt="persist_adam", lr_mult=1)}


def stats(p, y):
    tp = float(((p == 1) & (y == 1)).sum()); fp = float(((p == 1) & (y == 0)).sum()); fn = float(((p == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    return {"recall": rec, "precision": prec, "attack_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "alert_rate": float(p.mean())}


def main():
    p = get_pool("77")
    X17, y17, X18, y18, ts = p["X17"], p["y17"], p["X18"], p["y18"], p["ts18"]
    o = np.argsort(ts, kind="stable"); X18, y18 = X18[o], y18[o]
    benign = np.where(y18 == 0)[0]; attack = np.where(y18 == 1)[0]
    n_att = int(round(0.01 / 0.99 * len(benign)))
    out = {}
    for seed in SEEDS:
        torch.manual_seed(seed); m = pretrain_source_model(X17, y17, X17.shape[1])
        state = m.state_dict(); pre = score_buffer(m, X17, "logit")
        for d in range(DRAWS):
            rng = np.random.RandomState(seed * 100 + d)
            keep = np.sort(np.concatenate([benign, rng.choice(attack, n_att, replace=False)]))
            for order_name, idx in (("time_ordered", keep), ("shuffled", rng.permutation(keep))):
                for v, kw in V.items():
                    res = run_flex_trial(state, X17.shape[1], X18[idx], y18[idx], len(idx), space="logit",
                                         aqt=True, preseed=pre, **kw)
                    out.setdefault(order_name, {}).setdefault(v, []).append(stats(res["preds"], res["y"]))
            print(f"  seed {seed} draw {d} done", flush=True)
            json.dump(out, open("time_ordered_finetune_raw.json", "w"))


if __name__ == "__main__":
    main()
