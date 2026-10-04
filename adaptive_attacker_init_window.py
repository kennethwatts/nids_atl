"""
Adaptive-Attacker Experiment: Initialization-Window Abuse
===============================================================

Round 4 (Oct 4, 2026) Reviewer 3 item #2: "The threat model is
unchanged. Decoy inflation, pseudo-label poisoning, and initialization-
window abuse are untested. Section V is titled 'Dependability' but
contains no dependability evidence. Either run one small adaptive-
attacker experiment or retitle the section."

Section V's Threat Model already NAMES the W=500-sample AQT
initialization window as "the primary attack surface," during which
"sub-threshold bursts could evade detection before the buffer is
populated" -- but nothing in the paper tests this claim. This script
does, directly, for the first time.

Important clarification the paper's own text does not currently make
explicit: the "dead" tau=0.5 window only exists when AQT's buffer is
NOT pre-seeded (the no-pre-seed condition tested for Task #41). In the
paper's standard, pre-seeded configuration, the buffer already holds
500 source-domain predictions before the stream starts, so there is NO
tau=0.5 period at all -- the quantile is live from sample 1. The threat
model's "initialization window" attack surface is therefore specific to
deployments that skip source pre-seeding, and this experiment is run in
that (no-pre-seed) configuration to test the attack surface the paper
actually describes.

Two attacker models, same total malicious-sample BUDGET (40 malicious
samples in a 2,000-sample stream, i.e. 2% overall), same AQT-alone
configuration (no fine-tuning) with an EMPTY (non-pre-seeded) buffer:

1. "random_timing": the paper's existing i.i.d. attacker model --
   malicious samples scattered uniformly at random across all 2,000
   positions (what the paper already evaluates everywhere else).

2. "init_window_attack": an ADAPTIVE attacker that concentrates ALL 40
   malicious samples inside the first 500 positions specifically (the
   undefended tau=0.5 window), then sends only benign traffic for the
   remaining 1,500 positions.

Reports attack-class recall within the first-500-sample window
specifically (comparing the two attacker models head-to-head on exactly
the window the Threat Model names) and overall stream recall/F1, over
30 Monte Carlo runs.
"""

import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, recall_score

from verified_pipeline import (
    SEED, N_MONTE_CARLO, CONFIGS, AQT_WINDOW,
    run_one_trial, load_data, pretrain_source_model, build_source_pred_buffer,
)

torch.manual_seed(SEED)
np.random.seed(SEED)

AQT_CFG = CONFIGS["3_tl_aqt"]
STREAM_LEN = 2000
N_MALICIOUS = 40  # 2% overall budget, same for both attacker models
INIT_WINDOW = AQT_WINDOW  # 500


def build_attacker_stream(X18, y18, placement, rng):
    """placement: 'random' scatters N_MALICIOUS attack samples uniformly
    across all STREAM_LEN positions; 'init_window' concentrates all of
    them in positions [0, INIT_WINDOW)."""
    benign_idx = np.where(y18 == 0)[0]
    attack_idx = np.where(y18 == 1)[0]

    chosen_attack = rng.choice(attack_idx, N_MALICIOUS, replace=True)
    chosen_benign = rng.choice(benign_idx, STREAM_LEN - N_MALICIOUS, replace=True)

    X_stream = np.zeros((STREAM_LEN, X18.shape[1]), dtype=X18.dtype)
    y_stream = np.zeros(STREAM_LEN, dtype=y18.dtype)

    if placement == "random":
        positions = rng.choice(STREAM_LEN, N_MALICIOUS, replace=False)
    elif placement == "init_window":
        positions = rng.choice(INIT_WINDOW, N_MALICIOUS, replace=False)
    else:
        raise ValueError(placement)

    is_attack_pos = np.zeros(STREAM_LEN, dtype=bool)
    is_attack_pos[positions] = True

    X_stream[is_attack_pos] = X18[chosen_attack]
    y_stream[is_attack_pos] = 1.0
    X_stream[~is_attack_pos] = X18[chosen_benign]
    y_stream[~is_attack_pos] = 0.0

    return X_stream, y_stream


def main():
    print("Loading data, pretraining source model (identical to verified_pipeline.py)...")
    X17, y17, X18, y18, feature_cols = load_data()
    input_dim = X17.shape[1]
    model = pretrain_source_model(X17, y17, input_dim)
    pretrained_state = model.state_dict()
    # NOTE: deliberately NOT using build_source_pred_buffer here -- this experiment
    # specifically targets the no-pre-seed configuration (empty buffer), since that
    # is the configuration under which the Threat Model's "W=500 initialization
    # window" attack surface actually exists (see module docstring).

    results = {"random_timing": {"window_recall": [], "overall_recall": [], "overall_f1": []},
               "init_window_attack": {"window_recall": [], "overall_recall": [], "overall_f1": []}}

    t0 = time.time()
    for mc in range(N_MONTE_CARLO):
        for placement, key in [("random", "random_timing"), ("init_window", "init_window_attack")]:
            rng = np.random.RandomState(SEED + mc + (0 if placement == "random" else 500000))
            X_stream, y_stream = build_attacker_stream(X18, y18, placement, rng)
            preds_all, y_true = run_one_trial(
                "3_tl_aqt", AQT_CFG, pretrained_state, input_dim,
                X_stream, y_stream, [], STREAM_LEN  # empty buffer: no pre-seed
            )
            y_window, p_window = y_true[:INIT_WINDOW], preds_all[:INIT_WINDOW]
            window_recall = recall_score(y_window, p_window, pos_label=1, zero_division=0)
            overall_recall = recall_score(y_true, preds_all, pos_label=1, zero_division=0)
            overall_f1 = f1_score(y_true, preds_all, average="macro", zero_division=0)
            results[key]["window_recall"].append(float(window_recall))
            results[key]["overall_recall"].append(float(overall_recall))
            results[key]["overall_f1"].append(float(overall_f1))
        if (mc + 1) % 10 == 0:
            print(f"  MC run {mc+1}/{N_MONTE_CARLO} done ({time.time()-t0:.1f}s elapsed)")

    print(f"\n=== ADAPTIVE-ATTACKER EXPERIMENT: INIT-WINDOW ABUSE "
          f"(AQT-alone, no pre-seed, 30 MC runs, {N_MALICIOUS}/{STREAM_LEN} malicious budget) ===")
    rows = []
    for key in results:
        row = {"attacker_model": key}
        for metric in ["window_recall", "overall_recall", "overall_f1"]:
            vals = results[key][metric]
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals))
        rows.append(row)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    rand_window_recall = np.array(results["random_timing"]["window_recall"])
    init_window_recall = np.array(results["init_window_attack"]["window_recall"])
    print(f"\n  Init-window attacker's recall inside the first {INIT_WINDOW} samples: "
          f"{init_window_recall.mean():.4f} vs. random-timing attacker's: {rand_window_recall.mean():.4f}")

    df.to_csv("adaptive_attacker_init_window_results.csv", index=False)
    with open("adaptive_attacker_init_window_raw.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: adaptive_attacker_init_window_results.csv, adaptive_attacker_init_window_raw.json")


if __name__ == "__main__":
    main()
