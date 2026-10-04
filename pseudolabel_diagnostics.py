"""
Oracle Random-Skip Control + Pseudo-Label Precision Against Ground Truth
=============================================================================

Round 4 (Oct 4, 2026) Reviewer 2 item #2:
"Why pseudo-labels hurt less is unexplained. Attack-F1 is 0.038 for
pseudo-labels vs 0.003 for oracle. About 50% of samples are labeled
benign and 0.5% malicious, so the 100:1 one-sided pressure applies to
pseudo-labels too. The likely reason is simply fewer updates. Add an
oracle control that randomly skips 50% of updates. Report pseudo-label
precision against ground truth. With AUROC 0.585, 'confident' is a
within-buffer rank, not calibrated confidence."

Two parts:

1. MEASURE the pseudo-label mechanism's actual behavior (not assumed):
   what fraction of steps actually receive a pseudo-label update (vs.
   being skipped as ambiguous), and when a pseudo-label IS assigned, how
   often does it match the true label? This directly answers "report
   pseudo-label precision against ground truth" and gives the REAL
   update rate to use below instead of guessing "50%".

2. ORACLE RANDOM-SKIP CONTROL: reruns oracle fine-tuning (true labels,
   same reinitialized-per-step optimizer as the paper's main results,
   NOT the persistent-optimizer variant from Task #31/the Oct 1 rerun --
   this control is specifically about update COUNT, not optimizer
   state) but randomly skips updates at the SAME rate measured in (1),
   using a fresh independent random draw each step (not confidence-
   gated). If this control's F1 matches pseudo-label fine-tuning's F1,
   Reviewer 2's "it's just fewer updates" hypothesis is confirmed and
   pseudo-label gating's apparent safety has no confidence-calibration
   content. If the random-skip control still looks like full oracle
   (degrades below baseline) while pseudo-label does not, update COUNT
   alone does not explain the gap, and *which* updates are skipped
   (confidence-gated vs. random) matters.
"""

import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from verified_pipeline import (
    SEED, N_MONTE_CARLO, SAMPLE_CHECKPOINTS, CONFIGS, PHASE1_CUTOFF, BASE_LR,
    CompactMLP, sigmoid_weight, get_aqt_threshold, get_pseudo_bounds,
    load_data, pretrain_source_model, build_source_pred_buffer,
    make_cold_start_stream,
)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Part 1: pseudo-label diagnostics (update rate + precision vs ground truth)
# ---------------------------------------------------------------------------
def run_pseudo_trial_with_diagnostics(pretrained_state, input_dim, X_stream, y_stream,
                                       source_pred_buffer, n_samples):
    model = CompactMLP(input_dim=input_dim)
    model.load_state_dict(pretrained_state)
    buffer = list(source_pred_buffer)
    preds_all, y_true = [], []
    criterion = nn.BCELoss()

    n_updates = 0
    n_pos_pseudo, n_pos_pseudo_correct = 0, 0  # pseudo says malicious; is it?
    n_neg_pseudo, n_neg_pseudo_correct = 0, 0  # pseudo says benign; is it?

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

        tau_hi, tau_lo = get_pseudo_bounds(buffer)
        if tau_hi is None:
            label_for_step = None
        elif p_t > tau_hi:
            label_for_step = 1.0
        elif p_t < tau_lo:
            label_for_step = 0.0
        else:
            label_for_step = None

        if label_for_step is not None:
            n_updates += 1
            if label_for_step == 1.0:
                n_pos_pseudo += 1
                n_pos_pseudo_correct += int(y_t == 1.0)
            else:
                n_neg_pseudo += 1
                n_neg_pseudo_correct += int(y_t == 0.0)

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
            loss = criterion(model(x_t), target)
            loss.backward()
            optimizer.step()

    diag = dict(n_updates=n_updates, n_total=n_samples,
                n_pos_pseudo=n_pos_pseudo, n_pos_pseudo_correct=n_pos_pseudo_correct,
                n_neg_pseudo=n_neg_pseudo, n_neg_pseudo_correct=n_neg_pseudo_correct)
    return preds_all, y_true, diag


# ---------------------------------------------------------------------------
# Part 2: oracle random-skip control
# ---------------------------------------------------------------------------
def run_oracle_skip_trial(keep_prob, pretrained_state, input_dim, X_stream, y_stream,
                           source_pred_buffer, n_samples, skip_rng):
    model = CompactMLP(input_dim=input_dim)
    model.load_state_dict(pretrained_state)
    buffer = list(source_pred_buffer)
    preds_all, y_true = [], []
    criterion = nn.BCELoss()

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

        if skip_rng.random() < keep_prob:  # independent random draw, NOT confidence-gated
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
            target = torch.tensor([[y_t]], dtype=torch.float32)  # oracle: true label
            loss = criterion(model(x_t), target)
            loss.backward()
            optimizer.step()

    return preds_all, y_true


CHECKPOINT_PATH = "pseudolabel_diagnostics_checkpoint.json"


def load_checkpoint():
    try:
        with open(CHECKPOINT_PATH) as f:
            ck = json.load(f)
        pseudo_results = {int(n): v for n, v in ck["pseudo_results"].items()}
        skip_results = {int(n): v for n, v in ck["skip_results"].items()} if ck.get("skip_results") else None
        return ck["part1_mc_done"], pseudo_results, ck["agg"], ck.get("part2_mc_done", 0), skip_results
    except FileNotFoundError:
        return 0, {n: [] for n in SAMPLE_CHECKPOINTS}, dict(
            n_updates=0, n_total=0, n_pos_pseudo=0, n_pos_pseudo_correct=0,
            n_neg_pseudo=0, n_neg_pseudo_correct=0), 0, None


def save_checkpoint(part1_mc_done, pseudo_results, agg, part2_mc_done, skip_results):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({
            "part1_mc_done": part1_mc_done,
            "pseudo_results": {str(n): v for n, v in pseudo_results.items()},
            "agg": agg,
            "part2_mc_done": part2_mc_done,
            "skip_results": {str(n): v for n, v in skip_results.items()} if skip_results else None,
        }, f)


def main(time_budget_seconds=None):
    part1_mc_done, pseudo_results, agg, part2_mc_done, skip_results = load_checkpoint()
    if skip_results is None:
        skip_results = {n: [] for n in SAMPLE_CHECKPOINTS}

    print("Loading data, pretraining source model (identical to verified_pipeline.py)...")
    X17, y17, X18, y18, feature_cols = load_data()
    input_dim = X17.shape[1]
    model = pretrain_source_model(X17, y17, input_dim)
    pretrained_state = model.state_dict()
    source_pred_buffer = build_source_pred_buffer(model, X17)
    max_n = max(SAMPLE_CHECKPOINTS)
    t0 = time.time()

    # --- Part 1: pseudo-label diagnostics ---
    if part1_mc_done < N_MONTE_CARLO:
        print(f"\n--- Part 1: pseudo-label update rate + precision vs. ground truth "
              f"({part1_mc_done}/{N_MONTE_CARLO} done) ---")
        mc = part1_mc_done
        while mc < N_MONTE_CARLO:
            rng = np.random.RandomState(SEED + mc)
            X_stream, y_stream = make_cold_start_stream(X18, y18, max_n, 0.01, rng)
            preds_all, y_true, diag = run_pseudo_trial_with_diagnostics(
                pretrained_state, input_dim, X_stream, y_stream, source_pred_buffer, max_n
            )
            for n in SAMPLE_CHECKPOINTS:
                f1 = f1_score(y_true[:n], preds_all[:n], average="macro", zero_division=0)
                pseudo_results[n].append(f1)
            for k in agg:
                agg[k] += diag[k]
            mc += 1
            save_checkpoint(mc, pseudo_results, agg, part2_mc_done, skip_results)
            elapsed = time.time() - t0
            print(f"  [diagnostics] MC run {mc}/{N_MONTE_CARLO} done ({elapsed:.1f}s elapsed)", flush=True)
            if time_budget_seconds is not None and elapsed >= time_budget_seconds:
                print("Time budget reached (Part 1 incomplete). Re-run to continue.")
                return
        part1_mc_done = mc

    update_rate = agg["n_updates"] / agg["n_total"]
    pos_precision = agg["n_pos_pseudo_correct"] / agg["n_pos_pseudo"] if agg["n_pos_pseudo"] else float("nan")
    neg_precision = agg["n_neg_pseudo_correct"] / agg["n_neg_pseudo"] if agg["n_neg_pseudo"] else float("nan")

    print(f"\n  Pseudo-label update rate (fraction of steps NOT skipped as ambiguous): {update_rate:.4f}")
    print(f"  Positive pseudo-labels: {agg['n_pos_pseudo']} total, precision vs ground truth = {pos_precision:.4f}")
    print(f"  Negative pseudo-labels: {agg['n_neg_pseudo']} total, precision vs ground truth = {neg_precision:.4f}")

    # --- Part 2: oracle random-skip control, at the MEASURED update rate ---
    if part2_mc_done < N_MONTE_CARLO:
        print(f"\n--- Part 2: oracle random-skip control (keep_prob={update_rate:.4f}, "
              f"matched to measured pseudo-label update rate) ({part2_mc_done}/{N_MONTE_CARLO} done) ---")
        t0 = time.time()
        mc = part2_mc_done
        while mc < N_MONTE_CARLO:
            rng = np.random.RandomState(SEED + mc)
            X_stream, y_stream = make_cold_start_stream(X18, y18, max_n, 0.01, rng)
            skip_rng = np.random.RandomState(70000 + mc)
            preds_all, y_true = run_oracle_skip_trial(
                update_rate, pretrained_state, input_dim, X_stream, y_stream,
                source_pred_buffer, max_n, skip_rng
            )
            for n in SAMPLE_CHECKPOINTS:
                f1 = f1_score(y_true[:n], preds_all[:n], average="macro", zero_division=0)
                skip_results[n].append(f1)
            mc += 1
            save_checkpoint(part1_mc_done, pseudo_results, agg, mc, skip_results)
            elapsed = time.time() - t0
            print(f"  [oracle-skip] MC run {mc}/{N_MONTE_CARLO} done ({elapsed:.1f}s elapsed)", flush=True)
            if time_budget_seconds is not None and elapsed >= time_budget_seconds:
                print("Time budget reached (Part 2 incomplete). Re-run to continue.")
                return

    # Summarize against existing Table I numbers (oracle full, pseudo-label, base, AQT-alone)
    with open("verified_ablation_raw.json") as f:
        raw = json.load(f)

    rows = []
    for label, results_dict in [
        ("1_base_tl (paper)", {n: raw["1_base_tl"][str(n)] for n in SAMPLE_CHECKPOINTS}),
        ("3_tl_aqt (paper)", {n: raw["3_tl_aqt"][str(n)] for n in SAMPLE_CHECKPOINTS}),
        ("5a_full_atl_oracle, 100% updates (paper)", {n: raw["5a_full_atl_oracle"][str(n)] for n in SAMPLE_CHECKPOINTS}),
        (f"oracle_random_skip, keep_prob={update_rate:.3f} (this script)", skip_results),
        ("5b_full_atl_pseudo (paper, confidence-gated)", {n: raw["5b_full_atl_pseudo"][str(n)] for n in SAMPLE_CHECKPOINTS}),
    ]:
        row = {"condition": label}
        for n in SAMPLE_CHECKPOINTS:
            vals = results_dict[n]
            row[f"F1@{n}_mean"] = float(np.mean(vals))
            row[f"F1@{n}_std"] = float(np.std(vals))
        rows.append(row)
    df = pd.DataFrame(rows)

    print("\n=== ORACLE RANDOM-SKIP CONTROL vs. PAPER'S CONDITIONS (macro-F1, 30 MC runs) ===")
    print(df.to_string(index=False))

    df.to_csv("pseudolabel_diagnostics_comparison.csv", index=False)
    with open("pseudolabel_diagnostics_raw.json", "w") as f:
        json.dump({
            "update_rate": update_rate, "pos_precision": pos_precision, "neg_precision": neg_precision,
            "diagnostics_agg": agg,
            "pseudo_results_this_run": {str(n): v for n, v in pseudo_results.items()},
            "oracle_skip_results": {str(n): v for n, v in skip_results.items()},
        }, f, indent=2)
    print("\nSaved: pseudolabel_diagnostics_comparison.csv, pseudolabel_diagnostics_raw.json")


if __name__ == "__main__":
    import sys
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else None
    main(time_budget_seconds=budget)
