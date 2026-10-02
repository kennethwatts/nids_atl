"""
Persistent-Optimizer Oracle Rerun
====================================

Reviewer 2 (Round 2, reiterated Round 3 point 1) identifies that
verified_pipeline.py's oracle fine-tuning re-initializes the Adam optimizer
at every single adaptation step (see run_one_trial, lines ~269-275: a fresh
torch.optim.Adam(...) is constructed inside the per-sample loop). With no
persisted first/second-moment state, each update collapses to approximately
lr * sign(gradient) -- a full-size step regardless of gradient magnitude --
which is a plausible mechanical explanation for the oracle-fine-tuning
"collapse" reported in Section IV-C.2, independent of whether true-label
fine-tuning is unsafe in general.

This script reruns the oracle condition ("5a_full_atl_oracle": use_aqt=True,
adapt=True, pseudo_label=False) with the optimizer constructed ONCE per
trial and never re-initialized, for both Adam and SGD, across a grid of
4 learning rates, over the same 30 Monte Carlo cold-start streams used in
the paper. If the "oracle fine-tuning degrades below baseline" finding
survives with a persistent optimizer, it is a genuine property of naive
true-label fine-tuning under this imbalance regime, not an artifact of the
reinitialization bug. If it does not survive, Section IV-C.2's "mechanical
explanation" claim needs to become the paper's primary explanation instead
of a secondary one, and the abstract's framing needs another pass.

Reuses load_data / pretrain_source_model / build_source_pred_buffer /
make_cold_start_stream / CompactMLP / sigmoid_weight / get_aqt_threshold
verbatim from verified_pipeline.py -- only the optimizer lifecycle differs.
"""

import time
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from verified_pipeline import (
    SEED, N_MONTE_CARLO, SAMPLE_CHECKPOINTS, BASE_LR, PHASE1_CUTOFF,
    CompactMLP, sigmoid_weight, get_aqt_threshold,
    load_data, pretrain_source_model, build_source_pred_buffer,
    make_cold_start_stream,
)

LR_GRID = [BASE_LR * 10, BASE_LR, BASE_LR / 10, BASE_LR / 100]
OPTIMIZERS = {
    "adam_persistent": torch.optim.Adam,
    "sgd_persistent": torch.optim.SGD,
}

torch.manual_seed(SEED)
np.random.seed(SEED)


def run_persistent_oracle_trial(pretrained_state, input_dim, X_stream, y_stream,
                                 source_pred_buffer, n_samples, opt_name, opt_cls, lr):
    """
    Same AQT + discriminative-unfreezing + oracle-label logic as
    verified_pipeline.run_one_trial's "5a_full_atl_oracle" condition, except
    the optimizer is constructed once (persistent moment estimates) and its
    per-group learning rates are updated in place via param_groups rather
    than by re-instantiating the optimizer every step.
    """
    model = CompactMLP(input_dim=input_dim)
    model.load_state_dict(pretrained_state)

    buffer = list(source_pred_buffer)
    preds_all, y_true = [], []
    criterion = nn.BCELoss()

    # Two param groups from the start (output always active; hidden layers
    # are near-frozen via the sigmoid schedule's own near-zero LR early on,
    # rather than being excluded from the optimizer and added back later --
    # that exclude/re-include step is itself a form of state reset we avoid
    # here on purpose).
    optimizer = opt_cls([
        {"params": model.output_params(), "lr": lr},
        {"params": model.hidden_params(), "lr": 0.0},
    ])

    for t in range(1, n_samples + 1):
        x_t = torch.tensor(X_stream[t - 1:t], dtype=torch.float32)
        y_t = float(y_stream[t - 1])
        y_true.append(y_t)

        model.eval()
        with torch.no_grad():
            p_t = model(x_t).item()

        buffer.append(p_t)
        tau = get_aqt_threshold(buffer)
        preds_all.append(1 if p_t > tau else 0)

        label_for_step = y_t  # oracle: true label
        model.train()
        omega = sigmoid_weight(t)
        optimizer.param_groups[0]["lr"] = lr
        optimizer.param_groups[1]["lr"] = 0.0 if t <= PHASE1_CUTOFF else 0.1 * lr * omega

        optimizer.zero_grad()
        target = torch.tensor([[label_for_step]], dtype=torch.float32)
        loss = criterion(model(x_t), target)
        loss.backward()
        optimizer.step()

    return preds_all, y_true


CHECKPOINT_PATH = "persistent_optimizer_oracle_checkpoint.json"


def load_checkpoint():
    try:
        with open(CHECKPOINT_PATH) as f:
            ck = json.load(f)
        # JSON keys come back as strings; convert checkpoint sample-size keys back to int
        results = {
            key: {int(n): vals for n, vals in by_n.items()}
            for key, by_n in ck["results"].items()
        }
        return ck["completed_mc_runs"], results
    except FileNotFoundError:
        return 0, {
            f"{opt_name}_lr{lr:g}": {n: [] for n in SAMPLE_CHECKPOINTS}
            for opt_name in OPTIMIZERS for lr in LR_GRID
        }


def save_checkpoint(completed_mc_runs, results):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"completed_mc_runs": completed_mc_runs, "results": results}, f)


def write_final_outputs(results):
    rows = []
    for key, by_n in results.items():
        row = {"variant": key}
        for n in SAMPLE_CHECKPOINTS:
            vals = by_n[n]
            row[f"F1@{n}_mean"] = float(np.mean(vals))
            row[f"F1@{n}_std"] = float(np.std(vals))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("persistent_optimizer_oracle_results.csv", index=False)
    with open("persistent_optimizer_oracle_raw.json", "w") as f:
        json.dump(results, f, indent=2)
    return df


def main(time_budget_seconds=None):
    """
    Resumable: progress checkpoints to disk after every completed Monte Carlo
    run (persistent_optimizer_oracle_checkpoint.json), so a kill/restart of
    this process (e.g. a background job not surviving between tool-call
    boundaries) loses at most one partially-finished MC run, not the whole
    experiment. Re-running this script picks up where it left off.

    time_budget_seconds: if set, stop (and checkpoint) after roughly this
    much wall-clock time rather than running to completion, so this can be
    invoked repeatedly in bounded chunks.
    """
    completed_mc_runs, results = load_checkpoint()
    if completed_mc_runs >= N_MONTE_CARLO:
        print(f"Already complete: {completed_mc_runs}/{N_MONTE_CARLO} MC runs in checkpoint.")
        write_final_outputs(results)
        return

    print(f"Resuming from checkpoint: {completed_mc_runs}/{N_MONTE_CARLO} MC runs already done."
          if completed_mc_runs else "Starting fresh (no checkpoint found).")
    print("Loading data, pretraining source model (identical to verified_pipeline.py)...")
    X17, y17, X18, y18, feature_cols = load_data()
    input_dim = X17.shape[1]
    model = pretrain_source_model(X17, y17, input_dim)
    pretrained_state = model.state_dict()
    source_pred_buffer = build_source_pred_buffer(model, X17)

    max_n = max(SAMPLE_CHECKPOINTS)
    t0 = time.time()
    mc = completed_mc_runs
    while mc < N_MONTE_CARLO:
        rng = np.random.RandomState(SEED + mc)
        X_stream, y_stream = make_cold_start_stream(X18, y18, max_n, 0.01, rng)
        for opt_name, opt_cls in OPTIMIZERS.items():
            for lr in LR_GRID:
                key = f"{opt_name}_lr{lr:g}"
                preds_all, y_true = run_persistent_oracle_trial(
                    pretrained_state, input_dim, X_stream, y_stream,
                    source_pred_buffer, max_n, opt_name, opt_cls, lr
                )
                for n in SAMPLE_CHECKPOINTS:
                    f1 = f1_score(y_true[:n], preds_all[:n], average="macro", zero_division=0)
                    results[key][n].append(f1)
        mc += 1
        save_checkpoint(mc, results)
        elapsed = time.time() - t0
        print(f"  MC run {mc}/{N_MONTE_CARLO} done ({elapsed:.1f}s elapsed this invocation)", flush=True)
        if time_budget_seconds is not None and elapsed >= time_budget_seconds:
            print(f"Time budget ({time_budget_seconds}s) reached; checkpointed at "
                  f"{mc}/{N_MONTE_CARLO} MC runs. Re-run this script to continue.")
            return

    df = write_final_outputs(results)
    print(f"\n=== PERSISTENT-OPTIMIZER ORACLE RESULTS (macro-F1, mean +/- std, {N_MONTE_CARLO} MC runs) ===")
    print(df.to_string(index=False))
    print("\nSaved: persistent_optimizer_oracle_results.csv, persistent_optimizer_oracle_raw.json")


if __name__ == "__main__":
    import sys
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else None
    main(time_budget_seconds=budget)
