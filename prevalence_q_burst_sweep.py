"""
Prevalence Sweep, q Above 0.99, and Burst Scenario
=======================================================

Round 4 (Oct 4, 2026) Reviewer 2 item #3 and Reviewer 3 item #1;
Meta-Assessment priority #5 ("Prevalence sweep (0.5%, 2%, 5%, 20% burst)
with q up to 0.999").

Three related checks, all using the AQT-alone configuration (no
fine-tuning -- isolates AQT's own behavior from fine-tuning effects),
the same pretrained source model, and the same pre-seeded buffer
machinery as verified_pipeline.py / aqt_quantile_sensitivity.py:

1. PREVALENCE SWEEP (q fixed at the paper's q=0.99): cold-start F1 at
   malicious rates {0.5%, 1%, 2%, 5%} i.i.d. Reviewer 2: "q=0.99 equals
   the 1% prevalence... AQT effectively receives the true prevalence as
   an alert budget... Without this [sweep], 'AQT is the driver' may mean
   'knowing the prevalence is the driver'." If F1 degrades sharply once
   prevalence no longer matches q, that supports the prevalence-leakage
   concern; if it stays comparable, AQT's benefit is more robust to
   mismatched q than the single q=0.99=1%-prevalence case alone can show.

2. q ABOVE 0.99 (prevalence fixed at the paper's 1%): extends
   aqt_quantile_sensitivity.py's q in {0.80,...,0.99} sweep upward to
   q in {0.99, 0.995, 0.999}, to see whether F1 keeps rising past the
   paper's chosen value (in which case q=0.99 is not even locally
   optimal) or peaks there (consistent with, though not proof of, q=0.99
   tracking the true 1% prevalence specifically).

3. BURST SCENARIO: Reviewer 3: "A fixed-alert-rate threshold fails
   exactly when attacks are heavy. At q=0.99 the alert rate is about 1%
   by construction. In a DDoS or scan burst the buffer fills with attack
   scores, tau rises to match them, and recall collapses... Add a burst
   scenario (e.g. 20% malicious for 2,000 samples)." Constructs a stream
   that is 1% malicious (stable, AQT fully warmed up) for 10,000 samples,
   then jumps to 20% malicious for the next 2,000 samples, and reports
   attack-class recall separately for the pre-burst window (last 2,000
   baseline samples) and the burst window itself.

All three reuse verified_pipeline's pretraining, data loading, and
cold-start stream construction; none involves fine-tuning, so (unlike
the oracle/pseudo-label experiments) this should run to completion in a
single foreground invocation without needing checkpointing.
"""

import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, recall_score

from verified_pipeline import (
    SEED, N_MONTE_CARLO, SAMPLE_CHECKPOINTS, AQT_WINDOW, CONFIGS,
    run_one_trial, load_data, pretrain_source_model, build_source_pred_buffer,
    make_cold_start_stream,
)
import verified_pipeline as vp

torch.manual_seed(SEED)
np.random.seed(SEED)

AQT_CFG = CONFIGS["3_tl_aqt"]

PREVALENCE_GRID = [0.005, 0.01, 0.02, 0.05]
Q_FIXED = 0.99

Q_GRID_HIGH = [0.99, 0.995, 0.999]
PREVALENCE_FIXED = 0.01

BURST_BASELINE_N = 10000
BURST_N = 2000
BURST_RATE = 0.20
BURST_BASELINE_RATE = 0.01


def make_threshold_fn(q):
    """Same technique as aqt_quantile_sensitivity.py: get_aqt_threshold's
    default q is bound at import time, so we replace the function object
    verified_pipeline.get_aqt_threshold itself (run_one_trial looks it up
    from the module namespace at call time, not def time)."""
    def f(buffer, window=AQT_WINDOW):
        if len(buffer) < window:
            return 0.5
        return float(np.quantile(list(buffer)[-window:], q))
    return f


def run_prevalence_sweep(pretrained_state, input_dim, X18, y18, source_pred_buffer):
    results = {rate: {n: [] for n in SAMPLE_CHECKPOINTS} for rate in PREVALENCE_GRID}
    vp.get_aqt_threshold = make_threshold_fn(Q_FIXED)
    max_n = max(SAMPLE_CHECKPOINTS)
    for rate in PREVALENCE_GRID:
        for mc in range(N_MONTE_CARLO):
            rng = np.random.RandomState(SEED + mc)
            X_stream, y_stream = make_cold_start_stream(X18, y18, max_n, rate, rng)
            preds_all, y_true = run_one_trial(
                "3_tl_aqt", AQT_CFG, pretrained_state, input_dim,
                X_stream, y_stream, source_pred_buffer, max_n
            )
            for n in SAMPLE_CHECKPOINTS:
                f1 = f1_score(y_true[:n], preds_all[:n], average="macro", zero_division=0)
                results[rate][n].append(f1)
        print(f"  [prevalence sweep] rate={rate} done")
    return results


def run_q_high_sweep(pretrained_state, input_dim, X18, y18, source_pred_buffer):
    results = {q: {n: [] for n in SAMPLE_CHECKPOINTS} for q in Q_GRID_HIGH}
    max_n = max(SAMPLE_CHECKPOINTS)
    for q in Q_GRID_HIGH:
        vp.get_aqt_threshold = make_threshold_fn(q)
        for mc in range(N_MONTE_CARLO):
            rng = np.random.RandomState(SEED + mc)
            X_stream, y_stream = make_cold_start_stream(X18, y18, max_n, PREVALENCE_FIXED, rng)
            preds_all, y_true = run_one_trial(
                "3_tl_aqt", AQT_CFG, pretrained_state, input_dim,
                X_stream, y_stream, source_pred_buffer, max_n
            )
            for n in SAMPLE_CHECKPOINTS:
                f1 = f1_score(y_true[:n], preds_all[:n], average="macro", zero_division=0)
                results[q][n].append(f1)
        print(f"  [q>0.99 sweep] q={q} done")
    return results


def make_burst_stream(X18, y18, rng):
    X_base, y_base = make_cold_start_stream(X18, y18, BURST_BASELINE_N, BURST_BASELINE_RATE, rng)
    X_burst, y_burst = make_cold_start_stream(X18, y18, BURST_N, BURST_RATE, rng)
    X_stream = np.concatenate([X_base, X_burst], axis=0)
    y_stream = np.concatenate([y_base, y_burst], axis=0)
    return X_stream, y_stream


def run_burst_scenario(pretrained_state, input_dim, X18, y18, source_pred_buffer):
    vp.get_aqt_threshold = make_threshold_fn(Q_FIXED)  # restore paper's standard q=0.99
    pre_burst_recalls, burst_recalls = [], []
    pre_burst_alert_rates, burst_alert_rates = [], []
    pre_burst_f1s, burst_f1s = [], []
    total_n = BURST_BASELINE_N + BURST_N

    for mc in range(N_MONTE_CARLO):
        rng = np.random.RandomState(SEED + mc)
        X_stream, y_stream = make_burst_stream(X18, y18, rng)
        preds_all, y_true = run_one_trial(
            "3_tl_aqt", AQT_CFG, pretrained_state, input_dim,
            X_stream, y_stream, source_pred_buffer, total_n
        )
        # pre-burst reference window: the 2,000 samples immediately before the burst
        pre_slice = slice(BURST_BASELINE_N - BURST_N, BURST_BASELINE_N)
        burst_slice = slice(BURST_BASELINE_N, BURST_BASELINE_N + BURST_N)

        y_pre, p_pre = y_true[pre_slice], preds_all[pre_slice]
        y_burst, p_burst = y_true[burst_slice], preds_all[burst_slice]

        pre_burst_recalls.append(float(recall_score(y_pre, p_pre, pos_label=1, zero_division=0)))
        burst_recalls.append(float(recall_score(y_burst, p_burst, pos_label=1, zero_division=0)))
        pre_burst_alert_rates.append(float(np.mean(p_pre)))
        burst_alert_rates.append(float(np.mean(p_burst)))
        pre_burst_f1s.append(float(f1_score(y_pre, p_pre, average="macro", zero_division=0)))
        burst_f1s.append(float(f1_score(y_burst, p_burst, average="macro", zero_division=0)))
        if (mc + 1) % 10 == 0:
            print(f"  [burst scenario] MC run {mc+1}/{N_MONTE_CARLO} done")

    return {
        "pre_burst_recall_mean": float(np.mean(pre_burst_recalls)),
        "pre_burst_recall_std": float(np.std(pre_burst_recalls)),
        "burst_recall_mean": float(np.mean(burst_recalls)),
        "burst_recall_std": float(np.std(burst_recalls)),
        "pre_burst_alert_rate_mean": float(np.mean(pre_burst_alert_rates)),
        "burst_alert_rate_mean": float(np.mean(burst_alert_rates)),
        "pre_burst_f1_mean": float(np.mean(pre_burst_f1s)),
        "pre_burst_f1_std": float(np.std(pre_burst_f1s)),
        "burst_f1_mean": float(np.mean(burst_f1s)),
        "burst_f1_std": float(np.std(burst_f1s)),
    }


PARTIAL_PATH = "prevalence_q_burst_partial.json"


def save_partial(d):
    with open(PARTIAL_PATH, "w") as f:
        json.dump(d, f)


def load_partial():
    try:
        with open(PARTIAL_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def main(stage="all"):
    print("Loading data, pretraining source model (identical to verified_pipeline.py)...")
    X17, y17, X18, y18, feature_cols = load_data()
    input_dim = X17.shape[1]
    model = pretrain_source_model(X17, y17, input_dim)
    pretrained_state = model.state_dict()
    source_pred_buffer = build_source_pred_buffer(model, X17)

    partial = load_partial()
    t0 = time.time()

    if stage in ("all", "sweeps") and "prevalence_results" not in partial:
        print("\n--- Prevalence sweep (q=0.99 fixed) ---")
        prevalence_results = run_prevalence_sweep(pretrained_state, input_dim, X18, y18, source_pred_buffer)
        print(f"  done ({time.time()-t0:.1f}s elapsed)")
        partial["prevalence_results"] = {str(k): v for k, v in prevalence_results.items()}
        save_partial(partial)
    if stage in ("all", "sweeps") and "q_high_results" not in partial:
        print("\n--- q above 0.99 (prevalence=1% fixed) ---")
        q_high_results = run_q_high_sweep(pretrained_state, input_dim, X18, y18, source_pred_buffer)
        print(f"  done ({time.time()-t0:.1f}s elapsed)")
        partial["q_high_results"] = {str(k): v for k, v in q_high_results.items()}
        save_partial(partial)
    if stage in ("all", "burst") and "burst_results" not in partial:
        print("\n--- Burst scenario (1% baseline -> 20% burst for 2,000 samples) ---")
        burst_results = run_burst_scenario(pretrained_state, input_dim, X18, y18, source_pred_buffer)
        print(f"  done ({time.time()-t0:.1f}s elapsed)")
        partial["burst_results"] = burst_results
        save_partial(partial)

    if "prevalence_results" not in partial or "q_high_results" not in partial or "burst_results" not in partial:
        print("\nStage complete; other stages not yet run. Re-run with a different --stage to continue, "
              "or with no argument once all stages are in the partial file.")
        return

    # keys were stringified for JSON; convert back to float for the summary below
    prevalence_results = {float(k): {int(n): v for n, v in by_n.items()} for k, by_n in partial["prevalence_results"].items()}
    q_high_results = {float(k): {int(n): v for n, v in by_n.items()} for k, by_n in partial["q_high_results"].items()}
    burst_results = partial["burst_results"]

    # Summarize
    prev_rows = []
    for rate in PREVALENCE_GRID:
        row = {"prevalence": rate}
        for n in SAMPLE_CHECKPOINTS:
            vals = prevalence_results[rate][n]
            row[f"F1@{n}_mean"] = float(np.mean(vals))
            row[f"F1@{n}_std"] = float(np.std(vals))
        prev_rows.append(row)
    prev_df = pd.DataFrame(prev_rows)

    q_rows = []
    for q in Q_GRID_HIGH:
        row = {"q": q}
        for n in SAMPLE_CHECKPOINTS:
            vals = q_high_results[q][n]
            row[f"F1@{n}_mean"] = float(np.mean(vals))
            row[f"F1@{n}_std"] = float(np.std(vals))
        q_rows.append(row)
    q_df = pd.DataFrame(q_rows)

    print("\n=== PREVALENCE SWEEP (macro-F1, q=0.99 fixed, 30 MC runs) ===")
    print(prev_df.to_string(index=False))
    print("\n=== q ABOVE 0.99 (macro-F1, prevalence=1% fixed, 30 MC runs) ===")
    print(q_df.to_string(index=False))
    print("\n=== BURST SCENARIO (20% malicious for 2,000 samples after a 1%-baseline 10,000-sample warm-up) ===")
    for k, v in burst_results.items():
        print(f"  {k}: {v:.4f}")

    prev_df.to_csv("prevalence_sweep_results.csv", index=False)
    q_df.to_csv("q_above_099_results.csv", index=False)
    with open("burst_scenario_results.json", "w") as f:
        json.dump(burst_results, f, indent=2)
    with open("prevalence_q_burst_raw.json", "w") as f:
        json.dump({
            "prevalence_sweep": prevalence_results,
            "q_high_sweep": q_high_results,
            "burst_scenario": burst_results,
        }, f, indent=2)
    print("\nSaved: prevalence_sweep_results.csv, q_above_099_results.csv, "
          "burst_scenario_results.json, prevalence_q_burst_raw.json")


if __name__ == "__main__":
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    main(stage=stage)
