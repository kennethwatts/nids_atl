"""
Static-Quantile / Top-1%-Budget / No-Pre-Seed Baselines
============================================================

Round 4 (Oct 4, 2026 pre-submission review) Reviewer 2 item #3 and
Reviewer 3 item #3; named as Meta-Assessment priority #3 ("cheap... tests
whether 'adaptive' and 'pre-seeded' earn their names").

Three new conditions, all reusing verified_pipeline.py's model, data,
pretraining, and cold-start-stream construction (same 30 Monte Carlo
seeds as Table I, so directly comparable to it):

1. "static_source_q99": a threshold fixed ONCE, before any target-domain
   data is seen, at the 99th percentile of the frozen source-pretrained
   model's prediction distribution over the full 2017 (source) corpus.
   Applied as a fixed, never-updated threshold throughout the entire
   target stream -- no AQT buffer, no online recalibration at all. Tests
   whether AQT's ONLINE recalibration adds anything over simply knowing a
   good threshold from source data alone (Reviewer 2: "Compare against...
   a static source-benign 99th-percentile threshold").

2. "top1pct_budget": alert if p_t ranks in the top 1% of all scores seen
   so far in THIS stream (cumulative, unbounded, no W=500 window, no
   pre-seeding -- tau=0.5 for the first 100 samples until there is enough
   history for a meaningful quantile). This hands the model the TRUE
   malicious rate (1%) as a known alert budget and enforces it by rank,
   with none of AQT's specific windowed-quantile/pre-seed machinery.
   Tests Reviewer 2's concern that "AQT is the driver" may really mean
   "knowing the prevalence is the driver."

3. "aqt_no_preseed": identical to the paper's AQT-alone configuration
   (3_tl_aqt) EXCEPT the W=500 buffer starts EMPTY instead of pre-seeded
   with source-domain predictions, so tau=0.5 (the dead branch named in
   Algorithm 1 and in Section VII) is actually exercised for the stream's
   first 500 samples, which then self-populate the buffer. This is
   implemented for free by calling verified_pipeline.run_one_trial with
   source_pred_buffer=[] instead of the pre-seeded buffer -- no new logic
   needed, since run_one_trial already does
   `buffer = list(source_pred_buffer) if cfg["use_aqt"] else []`.
   This is the real cold-start pre-seed-vs-none test the paper's own
   Limitations section says it does not evaluate.

None of these three conditions involves any fine-tuning (no backward
pass), so this experiment is much cheaper than the oracle/pseudo-label
adaptation runs; it still checkpoints after every Monte Carlo run as a
safety margin, following the same resumable pattern used for the
persistent-optimizer rerun (backgrounded `nohup` processes do not
reliably survive across tool-call boundaries in this harness).
"""

import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from verified_pipeline import (
    SEED, N_MONTE_CARLO, SAMPLE_CHECKPOINTS, CONFIGS,
    CompactMLP, run_one_trial,
    load_data, pretrain_source_model, build_source_pred_buffer,
    make_cold_start_stream,
)

Q = 0.99
MIN_BUFFER_FOR_QUANTILE = 100  # top1pct_budget: tau=0.5 until this many samples seen

torch.manual_seed(SEED)
np.random.seed(SEED)


def compute_static_source_threshold(model, X17, q=Q):
    """99th-percentile threshold of the frozen source model's prediction
    distribution over the FULL source (2017) corpus -- computed once,
    before any target-domain data is touched, and never updated again."""
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X17, dtype=torch.float32)).numpy().flatten()
    return float(np.quantile(preds, q))


def run_static_threshold_trial(tau_fixed, pretrained_state, input_dim, X_stream, y_stream, n_samples):
    model = CompactMLP(input_dim=input_dim)
    model.load_state_dict(pretrained_state)
    model.eval()
    preds_all, y_true = [], []
    with torch.no_grad():
        for t in range(1, n_samples + 1):
            x_t = torch.tensor(X_stream[t - 1:t], dtype=torch.float32)
            y_t = float(y_stream[t - 1])
            y_true.append(y_t)
            p_t = model(x_t).item()
            preds_all.append(1 if p_t > tau_fixed else 0)
    return preds_all, y_true


def run_top1pct_budget_trial(pretrained_state, input_dim, X_stream, y_stream, n_samples,
                              min_buffer=MIN_BUFFER_FOR_QUANTILE, q=Q):
    model = CompactMLP(input_dim=input_dim)
    model.load_state_dict(pretrained_state)
    model.eval()
    preds_all, y_true = [], []
    scores_seen = []
    with torch.no_grad():
        for t in range(1, n_samples + 1):
            x_t = torch.tensor(X_stream[t - 1:t], dtype=torch.float32)
            y_t = float(y_stream[t - 1])
            y_true.append(y_t)
            p_t = model(x_t).item()
            scores_seen.append(p_t)
            tau = 0.5 if len(scores_seen) < min_buffer else float(np.quantile(scores_seen, q))
            preds_all.append(1 if p_t > tau else 0)
    return preds_all, y_true


CONDITIONS = ["static_source_q99", "top1pct_budget", "aqt_no_preseed"]
CHECKPOINT_PATH = "static_and_budget_baselines_checkpoint.json"


def load_checkpoint():
    try:
        with open(CHECKPOINT_PATH) as f:
            ck = json.load(f)
        results = {
            key: {int(n): vals for n, vals in by_n.items()}
            for key, by_n in ck["results"].items()
        }
        return ck["completed_mc_runs"], ck["tau_fixed"], results
    except FileNotFoundError:
        return 0, None, {key: {n: [] for n in SAMPLE_CHECKPOINTS} for key in CONDITIONS}


def save_checkpoint(completed_mc_runs, tau_fixed, results):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"completed_mc_runs": completed_mc_runs, "tau_fixed": tau_fixed, "results": results}, f)


def write_final_outputs(results, tau_fixed):
    rows = []
    for key in CONDITIONS:
        by_n = results[key]
        row = {"condition": key}
        for n in SAMPLE_CHECKPOINTS:
            vals = by_n[n]
            row[f"F1@{n}_mean"] = float(np.mean(vals))
            row[f"F1@{n}_std"] = float(np.std(vals))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("static_and_budget_baselines_results.csv", index=False)
    with open("static_and_budget_baselines_raw.json", "w") as f:
        json.dump({"tau_fixed_source_q99": tau_fixed, "results": results}, f, indent=2)
    return df


def main(time_budget_seconds=None):
    completed_mc_runs, tau_fixed, results = load_checkpoint()
    if completed_mc_runs >= N_MONTE_CARLO:
        print(f"Already complete: {completed_mc_runs}/{N_MONTE_CARLO} MC runs in checkpoint.")
        write_final_outputs(results, tau_fixed)
        return

    print(f"Resuming from checkpoint: {completed_mc_runs}/{N_MONTE_CARLO} MC runs already done."
          if completed_mc_runs else "Starting fresh (no checkpoint found).")
    print("Loading data, pretraining source model (identical to verified_pipeline.py)...")
    X17, y17, X18, y18, feature_cols = load_data()
    input_dim = X17.shape[1]
    model = pretrain_source_model(X17, y17, input_dim)
    pretrained_state = model.state_dict()
    source_pred_buffer = build_source_pred_buffer(model, X17)

    if tau_fixed is None:
        tau_fixed = compute_static_source_threshold(model, X17)
        print(f"  Static source-q99 threshold (fixed for the whole run): {tau_fixed:.4f}")
    else:
        print(f"  Static source-q99 threshold (from checkpoint): {tau_fixed:.4f}")

    aqt_cfg = CONFIGS["3_tl_aqt"]
    max_n = max(SAMPLE_CHECKPOINTS)
    t0 = time.time()
    mc = completed_mc_runs
    while mc < N_MONTE_CARLO:
        rng = np.random.RandomState(SEED + mc)
        X_stream, y_stream = make_cold_start_stream(X18, y18, max_n, 0.01, rng)

        preds_a, y_a = run_static_threshold_trial(tau_fixed, pretrained_state, input_dim, X_stream, y_stream, max_n)
        preds_b, y_b = run_top1pct_budget_trial(pretrained_state, input_dim, X_stream, y_stream, max_n)
        preds_c, y_c = run_one_trial("3_tl_aqt", aqt_cfg, pretrained_state, input_dim,
                                      X_stream, y_stream, [], max_n)  # empty buffer = no pre-seed

        for key, (preds_all, y_true) in zip(CONDITIONS, [(preds_a, y_a), (preds_b, y_b), (preds_c, y_c)]):
            for n in SAMPLE_CHECKPOINTS:
                f1 = f1_score(y_true[:n], preds_all[:n], average="macro", zero_division=0)
                results[key][n].append(f1)

        mc += 1
        save_checkpoint(mc, tau_fixed, results)
        elapsed = time.time() - t0
        print(f"  MC run {mc}/{N_MONTE_CARLO} done ({elapsed:.1f}s elapsed this invocation)", flush=True)
        if time_budget_seconds is not None and elapsed >= time_budget_seconds:
            print(f"Time budget ({time_budget_seconds}s) reached; checkpointed at "
                  f"{mc}/{N_MONTE_CARLO} MC runs. Re-run this script to continue.")
            return

    df = write_final_outputs(results, tau_fixed)
    print(f"\n=== STATIC/BUDGET/NO-PRESEED BASELINES (macro-F1, mean +/- std, {N_MONTE_CARLO} MC runs) ===")
    print(df.to_string(index=False))
    print(f"\nFixed source-q99 threshold used throughout: {tau_fixed:.4f}")
    print("Saved: static_and_budget_baselines_results.csv, static_and_budget_baselines_raw.json")


if __name__ == "__main__":
    import sys
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else None
    main(time_budget_seconds=budget)
