"""
Quantile-on-Logits Fix + Infiltration Rerun
================================================

Round 4 (Oct 4, 2026) Reviewer 3 item #4, Meta-Assessment priority #4:
"Score saturation is still an avoidable failure. Apply the quantile to
logits, or handle ties with >=. This likely removes the near-silent
infiltration behavior and the bit-identical results."

Section IV-D (infiltration_case_study.py) found that under the severe
Infiltration domain shift, 97% of raw sigmoid outputs are floating-point-
exact 1.0, so AQT's quantile buffer saturates and its threshold converges
to 1.0, making the classifier almost silent (5 alerts of 7,200) and
making oracle/pseudo-label fine-tuning bit-for-bit identical to AQT-alone
across all 30 runs (there is nothing left for fine-tuning to change once
almost nothing clears the saturated threshold).

This script reruns that same case study with ONE change: AQT's buffer,
threshold, and pseudo-label bounds all operate on the model's PRE-SIGMOID
LOGIT instead of the bounded [0,1] probability. Logits are unbounded, so
even samples that saturate to probability 1.0 remain distinguishable by
magnitude in logit space (a logit of 12 and a logit of 40 are both
sigmoid(.) = 1.0 to float32 precision, but very different as logits).
Everything else -- source pretraining, the 10-feature pipeline, the
cold-start stream construction, the four ablation configs, 30 Monte
Carlo runs, the 7,200-sample window -- is identical to
infiltration_case_study.py, so the two can be compared directly.

If the logit-space threshold stops saturating, we expect: alert rate
closer to the AQT-alone configuration's "intended" ~1% rather than
0.07%, and oracle/pseudo-label fine-tuning no longer bit-identical to
AQT-alone (since there is now something for fine-tuning to act on).
"""

import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from verified_pipeline import (
    CompactMLP, SEED, N_MONTE_CARLO, AQT_WINDOW, PHASE1_CUTOFF,
    BASE_LR, PSEUDO_HI_QUANTILE, PSEUDO_LO_QUANTILE, sigmoid_weight,
)
from infiltration_case_study import (
    load_data, make_infiltration_stream, TARGET_MALICIOUS_RATE,
    COLDSTART_WINDOW, DATA_DIR,
)
from verified_pipeline import pretrain_source_model

CONFIGS = {
    "1_base_tl":          dict(use_aqt=False, adapt=False, pseudo_label=False),
    "3_tl_aqt_logit":      dict(use_aqt=True,  adapt=False, pseudo_label=False),
    "5a_full_atl_oracle_logit": dict(use_aqt=True,  adapt=True,  pseudo_label=False),
    "5b_full_atl_pseudo_logit": dict(use_aqt=True,  adapt=True,  pseudo_label=True),
}

torch.manual_seed(SEED)
np.random.seed(SEED)


def model_logit(model, x):
    """Pre-sigmoid logit: same forward pass as CompactMLP.forward() but
    stopping before the final Sigmoid (model.out[0] is the Linear layer,
    model.out[1] is the Sigmoid)."""
    h = model.block2(model.block1(x))
    return model.out[0](h)


def build_source_logit_buffer(model, X17, window=AQT_WINDOW, seed=SEED):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X17), window, replace=False)
    model.eval()
    with torch.no_grad():
        logits = model_logit(model, torch.tensor(X17[idx], dtype=torch.float32)).numpy().flatten()
    return logits.tolist()


def get_logit_threshold(buffer, q=0.99, default_tau=0.0, window=AQT_WINDOW):
    """Same quantile-threshold logic as verified_pipeline.get_aqt_threshold,
    but operating on logits; default_tau=0.0 is the logit-space equivalent
    of a 0.5 probability threshold (sigmoid(0)=0.5), so the "dead branch"
    before the buffer fills behaves identically to the paper's version."""
    if len(buffer) < window:
        return default_tau
    return float(np.quantile(list(buffer)[-window:], q))


def get_logit_pseudo_bounds(buffer, window=AQT_WINDOW):
    if len(buffer) < window:
        return None, None
    recent = list(buffer)[-window:]
    logit_hi = float(np.quantile(recent, PSEUDO_HI_QUANTILE))
    logit_lo = float(np.quantile(recent, PSEUDO_LO_QUANTILE))
    return logit_hi, logit_lo


def run_one_trial_logit(cfg_id, cfg, pretrained_state, input_dim,
                         X_stream, y_stream, source_logit_buffer, n_samples):
    model = CompactMLP(input_dim=input_dim)
    model.load_state_dict(pretrained_state)

    buffer = list(source_logit_buffer) if cfg["use_aqt"] else []
    preds_all, y_true = [], []
    criterion = nn.BCELoss()

    for t in range(1, n_samples + 1):
        x_t = torch.tensor(X_stream[t - 1:t], dtype=torch.float32)
        y_t = float(y_stream[t - 1])
        y_true.append(y_t)

        model.eval()
        with torch.no_grad():
            logit_t = model_logit(model, x_t).item()

        if cfg["use_aqt"]:
            buffer.append(logit_t)
            tau = get_logit_threshold(buffer)
        else:
            tau = 0.0  # logit-space default (sigmoid(0) = 0.5)
        preds_all.append(1 if logit_t > tau else 0)

        if cfg["adapt"]:
            if cfg["pseudo_label"]:
                logit_hi, logit_lo = get_logit_pseudo_bounds(buffer)
                if logit_hi is None:
                    label_for_step = None
                elif logit_t > logit_hi:
                    label_for_step = 1.0
                elif logit_t < logit_lo:
                    label_for_step = 0.0
                else:
                    label_for_step = None
            else:
                label_for_step = y_t

            if label_for_step is not None:
                model.train()
                omega = sigmoid_weight(t)
                if t <= PHASE1_CUTOFF:
                    optimizer = torch.optim.Adam(model.output_params(), lr=BASE_LR)
                else:
                    optimizer = torch.optim.Adam([
                        {"params": model.output_params(), "lr": BASE_LR},
                        {"params": model.hidden_params(), "lr": 0.1 * BASE_LR * omega},
                    ])
                optimizer.zero_grad()
                target = torch.tensor([[label_for_step]], dtype=torch.float32)
                loss = criterion(model(x_t), target)  # BCELoss on the sigmoid output, as in the paper
                loss.backward()
                optimizer.step()

    return preds_all, y_true


CHECKPOINT_PATH = "logit_quantile_infiltration_checkpoint.json"


def load_checkpoint():
    try:
        with open(CHECKPOINT_PATH) as f:
            ck = json.load(f)
        return ck["completed_mc_runs"], ck["results"], ck["alert_rates"]
    except FileNotFoundError:
        return 0, {cfg_id: [] for cfg_id in CONFIGS}, {cfg_id: [] for cfg_id in CONFIGS}


def save_checkpoint(completed_mc_runs, results, alert_rates):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"completed_mc_runs": completed_mc_runs, "results": results, "alert_rates": alert_rates}, f)


def main(time_budget_seconds=None):
    completed_mc_runs, results, alert_rates = load_checkpoint()
    if completed_mc_runs >= N_MONTE_CARLO:
        print(f"Already complete: {completed_mc_runs}/{N_MONTE_CARLO} MC runs in checkpoint.")
    else:
        print(f"Resuming from checkpoint: {completed_mc_runs}/{N_MONTE_CARLO} MC runs already done."
              if completed_mc_runs else "Starting fresh (no checkpoint found).")
        print("Loading source (2017, deduped) + infiltration-only target, identical to infiltration_case_study.py...")
        X17, y17, Xi, yi, feature_cols = load_data()
        input_dim = X17.shape[1]
        print(f"  target (infiltration-only, deduped): {len(Xi)} rows, {100*yi.mean():.1f}% infiltration")
        print(f"  cold-start window: {COLDSTART_WINDOW} samples @ {100*TARGET_MALICIOUS_RATE:.1f}% malicious, {N_MONTE_CARLO} MC runs")

        print("\nPretraining source-domain model on 2017 data (identical seed/epochs to verified_pipeline.py)...")
        model = pretrain_source_model(X17, y17, input_dim)
        pretrained_state = model.state_dict()
        source_logit_buffer = build_source_logit_buffer(model, X17)
        print(f"  source logit buffer: min={min(source_logit_buffer):.2f}, max={max(source_logit_buffer):.2f}, "
              f"q99={np.quantile(source_logit_buffer, 0.99):.2f}")

        t0 = time.time()
        mc = completed_mc_runs
        while mc < N_MONTE_CARLO:
            rng = np.random.RandomState(SEED + mc)
            X_stream, y_stream = make_infiltration_stream(Xi, yi, COLDSTART_WINDOW, TARGET_MALICIOUS_RATE, rng)
            for cfg_id, cfg in CONFIGS.items():
                preds_all, y_true = run_one_trial_logit(
                    cfg_id, cfg, pretrained_state, input_dim,
                    X_stream, y_stream, source_logit_buffer, COLDSTART_WINDOW
                )
                f1 = f1_score(y_true, preds_all, average="macro", zero_division=0)
                results[cfg_id].append(f1)
                alert_rates[cfg_id].append(float(np.mean(preds_all)))
            mc += 1
            save_checkpoint(mc, results, alert_rates)
            elapsed = time.time() - t0
            print(f"  MC run {mc}/{N_MONTE_CARLO} done ({elapsed:.1f}s elapsed this invocation)", flush=True)
            if time_budget_seconds is not None and elapsed >= time_budget_seconds:
                print(f"Time budget ({time_budget_seconds}s) reached; checkpointed at "
                      f"{mc}/{N_MONTE_CARLO} MC runs. Re-run this script to continue.")
                return

    print("\n=== LOGIT-QUANTILE INFILTRATION CASE STUDY (macro-F1, mean +/- std, 30 runs, n=7200) ===")
    summary = {}
    for cfg_id in CONFIGS:
        vals = results[cfg_id]
        ar = alert_rates[cfg_id]
        summary[cfg_id] = {
            "f1_mean": float(np.mean(vals)), "f1_std": float(np.std(vals)),
            "alert_rate_mean": float(np.mean(ar)), "alert_rate_std": float(np.std(ar)),
        }
        print(f"  {cfg_id:28s}  F1 = {np.mean(vals):.3f} +/- {np.std(vals):.3f}   "
              f"alert-rate = {np.mean(ar)*100:.2f}%")

    # Bit-identical check: are oracle/pseudo-label runs identical to AQT-alone per MC run,
    # as they were (across all 30 runs) in the original probability-space version?
    oracle_vs_aqt_identical = sum(
        1 for a, b in zip(results["5a_full_atl_oracle_logit"], results["3_tl_aqt_logit"]) if a == b
    )
    pseudo_vs_aqt_identical = sum(
        1 for a, b in zip(results["5b_full_atl_pseudo_logit"], results["3_tl_aqt_logit"]) if a == b
    )
    print(f"\n  Oracle == AQT-alone (logit space): {oracle_vs_aqt_identical}/{N_MONTE_CARLO} MC runs bit-identical")
    print(f"  Pseudo == AQT-alone (logit space): {pseudo_vs_aqt_identical}/{N_MONTE_CARLO} MC runs bit-identical")
    summary["_diagnostics"] = {
        "source_logit_buffer_q99": float(np.quantile(source_logit_buffer, 0.99)),
        "oracle_vs_aqt_identical_runs": oracle_vs_aqt_identical,
        "pseudo_vs_aqt_identical_runs": pseudo_vs_aqt_identical,
        "n_mc_runs": N_MONTE_CARLO,
    }

    with open("logit_quantile_infiltration_raw.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("logit_quantile_infiltration_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved: logit_quantile_infiltration_raw.json, logit_quantile_infiltration_results.json")


if __name__ == "__main__":
    import sys
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else None
    main(time_budget_seconds=budget)
