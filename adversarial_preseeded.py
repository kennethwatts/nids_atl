"""
Attacks against the PRE-SEEDED default configuration (Round 5 Reviewer 3
major concerns 1-2).

Reviewer 3: the only attacker test targeted the no-pre-seed configuration the
paper does not use, and showed an effect any attacker gets by sending traffic
early. "Decoy inflation and pseudo-label poisoning against pre-seeded AQT are
the obvious experiments. A top-1% rule needs only about 1% decoys in the
buffer to shift tau. Add a decoy-fraction sweep, and a slow-drift poisoning
run."

Part A, decoy inflation (frozen model, so cheap). The attacker adds decoys:
flows that are benign in ground truth but score in the top 1% of benign
traffic under the frozen model, injected at a uniform random rate d. They
inflate the AQT buffer's upper quantile so that REAL attacks (1% of the
stream, unchanged) fall below the threshold. Compared rules: aqt (the paper),
aqt_cap (threshold capped at the source-benign 99th percentile: a hybrid that
cannot be pushed above what the source domain supports), src_q99 (static),
budget_1pct (causal cumulative top-1%). Reported: recall on real attacks,
precision, alert rate, as d is swept over {0, 0.25, 0.5, 1, 2, 5, 10}%.

Part B, slow-drift pseudo-label poisoning (fine-tuning in the loop). The
attacker injects "stealthy" attack flows at rate p: attack-class rows whose
frozen score is below the benign median, so the confidence gate labels them
confident-benign and the model is fine-tuned to treat attack-like traffic as
benign. Reported: recall on the REGULAR attacks (1% of the stream, not the
poison) over the last 10,000 samples, for pseudo-label fine-tuning with the
reset and persistent optimizers, against an AQT-alone control (no learning)
and oracle fine-tuning (poison correctly labeled attack).

Scoring pools: raw77 (logit space, a strong base model, so evasion is
measurable) for both parts; Part A also on paper10 (probability space).

usage: python3 adversarial_preseeded.py A <paper10|raw77>
       python3 adversarial_preseeded.py B <shard> <nshards>
"""

import json
import sys

import numpy as np
import torch

from verified_pipeline import pretrain_source_model
from flex_trial import model_logit, score_buffer, run_flex_trial
from grid_runner import load_pool
from threshold_baselines import scores_of, aqt_preds, budget_preds

N = 20000
DECOY_RATES = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10]
POISON_RATES = [0.0, 0.01, 0.02, 0.05]


def real_attack_metrics(preds, y, mask_real=None):
    m = np.ones(len(y), dtype=bool) if mask_real is None else mask_real
    att = (y == 1) & m
    recall = float(preds[att].mean()) if att.any() else float("nan")
    tp = float((preds[att] == 1).sum()); fp = float(((preds == 1) & (y == 0)).sum())
    return {"recall": recall, "precision": tp / (tp + fp) if tp + fp > 0 else 0.0,
            "alert_rate": float(preds.mean())}


def part_a(pool):
    space = "logit" if pool == "raw77" else "prob"
    default_tau = 0.0 if space == "logit" else 0.5
    X17, y17, X18, y18 = load_pool(pool)
    dim = X17.shape[1]
    out = {}
    for seed in [42, 123, 456]:
        torch.manual_seed(seed)
        m = pretrain_source_model(X17, y17, dim)
        s17 = scores_of(m, X17, space)
        src_q99 = float(np.quantile(s17[y17 == 0], 0.99))
        pre = score_buffer(m, X17, space)
        s18 = scores_of(m, X18, space)
        benign_idx = np.where(y18 == 0)[0]; attack_idx = np.where(y18 == 1)[0]
        b_scores = s18[benign_idx]
        decoy_pool = benign_idx[b_scores >= np.quantile(b_scores, 0.99)]
        for j in range(8):
            for d in DECOY_RATES:
                rng = np.random.RandomState(seed * 1000 + j)
                n_att = int(round(N * 0.01)); n_dec = int(round(N * d))
                n_ben = N - n_att - n_dec
                idx = np.concatenate([rng.choice(benign_idx, n_ben), rng.choice(attack_idx, n_att),
                                      rng.choice(decoy_pool, n_dec) if n_dec else np.array([], dtype=int)])
                rng.shuffle(idx)
                s = s18[idx]; ys = y18[idx]
                rules = {"aqt": aqt_preds(s, pre, 500, 0.99, default_tau),
                         "aqt_cap": aqt_preds(s, pre, 500, 0.99, default_tau, cap=src_q99),
                         "src_q99": (s > src_q99).astype(np.int8),
                         "budget_1pct": budget_preds(s, 0.01, default_tau)}
                for r, p in rules.items():
                    out.setdefault(f"{seed}_{j}_{d}", {})[r] = real_attack_metrics(p, ys)
        print(f"  Part A {pool}: seed {seed} done", flush=True)
    json.dump(out, open(f"adversarial_decoy_{pool}_raw.json", "w"))


def build_poison_stream(X18, y18, s18, p, rng):
    benign_idx = np.where(y18 == 0)[0]; attack_idx = np.where(y18 == 1)[0]
    med_b = np.median(s18[benign_idx])
    stealth = attack_idx[s18[attack_idx] < med_b]
    if len(stealth) < 20:  # fall back to the lowest-scoring 10% of attack rows
        stealth = attack_idx[np.argsort(s18[attack_idx])[:max(20, len(attack_idx) // 10)]]
    n_reg = int(round(N * 0.01)); n_poi = int(round(N * p)); n_ben = N - n_reg - n_poi
    idx = np.concatenate([rng.choice(benign_idx, n_ben), rng.choice(attack_idx, n_reg),
                          rng.choice(stealth, n_poi) if n_poi else np.array([], dtype=int)])
    is_poison = np.concatenate([np.zeros(n_ben + n_reg, dtype=bool), np.ones(n_poi, dtype=bool)])
    perm = rng.permutation(N)
    return X18[idx][perm], y18[idx][perm], is_poison[perm], len(stealth)


def part_b(shard, nshards):
    X17, y17, X18, y18 = load_pool("raw77")
    dim = X17.shape[1]
    torch.manual_seed(42)
    m = pretrain_source_model(X17, y17, dim)
    state = m.state_dict()
    pre = score_buffer(m, X17, "logit")
    s18 = scores_of(m, X18, "logit")
    variants = {
        "aqt_alone": dict(adapt="none"),
        "oracle_reset": dict(adapt="oracle", opt="reset_adam"),
        "pseudo_reset": dict(adapt="pseudo", opt="reset_adam"),
        "pseudo_adam1x": dict(adapt="pseudo", opt="persist_adam", lr_mult=1),
    }
    path = f"adversarial_poison_shard{shard}.json"
    try:
        out = json.load(open(path))
    except FileNotFoundError:
        out = {}
    units = [(j, p) for j in range(12) for p in POISON_RATES]
    for u, (j, p) in enumerate(units):
        key = f"{j}_{p}"
        if u % nshards != shard or key in out:
            continue
        rng = np.random.RandomState(42000 + j)
        Xs, ys, is_poi, n_stealth = build_poison_stream(X18, y18, s18, p, rng)
        rec = {"n_stealth_pool": int(n_stealth)}
        for v, kw in variants.items():
            res = run_flex_trial(state, dim, Xs, ys, N, space="logit", aqt=True, preseed=pre, **kw)
            pr, yy = res["preds"], res["y"]
            tail = np.arange(N) >= N - 10000
            reg = ~is_poi
            rec[v] = {"recall_regular_last10k": real_attack_metrics(pr[tail], yy[tail], reg[tail])["recall"],
                      "recall_regular_all": real_attack_metrics(pr, yy, reg)["recall"],
                      "recall_poison_all": float(pr[is_poi & (yy == 1)].mean()) if (is_poi).any() else float("nan"),
                      "alert_rate": float(pr.mean())}
        out[key] = rec
        json.dump(out, open(path, "w"))
        print(f"  Part B unit stream={j} poison={p} done", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "A":
        part_a(sys.argv[2])
    else:
        part_b(int(sys.argv[2]), int(sys.argv[3]))
