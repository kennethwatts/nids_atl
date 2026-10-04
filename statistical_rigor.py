"""
Statistical Rigor Pass: Effect Sizes, Multiple-Comparison Correction,
Pretraining-Seed Variation, and Pool Bootstrap
==========================================================================

Round 4 (Oct 4, 2026) Reviewer 2 item #4:
"The p-values describe resampling noise. There is one pretrained model,
one pool, and 30 resamples, so p can be made arbitrarily small. There
are no effect sizes or CIs, and 15 tests are uncorrected. The oracle
p=0.016 and p=0.011 would not survive correction. Vary the pretraining
seed and bootstrap over the pool."

Three stages (run with --stage effect_sizes | seeds | bootstrap | all):

1. EFFECT SIZES + CORRECTION (no new model runs): recomputes Table II's
   5 comparisons x 3 checkpoints = 15 paired t-tests directly from the
   already-saved verified_ablation_raw.json, adds paired Cohen's d for
   each, and applies Holm-Bonferroni correction across all 15 p-values
   to see which survive.

2. PRETRAINING-SEED VARIATION: reruns Base TL vs. AQT-alone (the paper's
   primary, load-bearing comparison; no fine-tuning, so this is cheap)
   across 5 different pretraining seeds instead of just SEED=42, each
   with the standard 30 Monte Carlo cold-start streams. Tests whether
   the AQT advantage is a property of the method or an artifact of one
   particular pretrained model.

3. POOL BOOTSTRAP: resamples the deduplicated target (2018) pool's row
   INDICES with replacement to build B bootstrap variants of the pool
   itself (not just the malicious-rate stream, which the paper's 30 MC
   runs already resample) -- addressing the reviewer's specific point
   that 30 MC resamples from ONE fixed pool cannot capture pool-level
   sampling uncertainty. One cold-start stream per bootstrap pool,
   Base TL vs. AQT-alone, giving a percentile bootstrap CI on the F1
   difference that accounts for pool composition itself being uncertain.
"""

import json
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import f1_score

from verified_pipeline import (
    SEED, N_MONTE_CARLO, SAMPLE_CHECKPOINTS, CONFIGS,
    run_one_trial, load_data, pretrain_source_model, build_source_pred_buffer,
    make_cold_start_stream, DATA_DIR,
)

BASE_CFG = CONFIGS["1_base_tl"]
AQT_CFG = CONFIGS["3_tl_aqt"]

COMPARISONS = [
    ("3_tl_aqt", "1_base_tl"),
    ("5a_full_atl_oracle", "1_base_tl"),
    ("5b_full_atl_pseudo", "1_base_tl"),
    ("5b_full_atl_pseudo", "5a_full_atl_oracle"),
    ("5b_full_atl_pseudo", "3_tl_aqt"),
]
SIG_CHECKPOINTS = [5000, 10000, 20000]


# ---------------------------------------------------------------------------
# Stage 1: effect sizes + Holm-Bonferroni correction on existing results
# ---------------------------------------------------------------------------
def cohens_d_paired(a, b):
    diff = np.array(a) - np.array(b)
    return float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0.0


def holm_bonferroni(pvals):
    """Returns adjusted significance (reject/accept at alpha=0.05) per the
    Holm step-down procedure, and the Holm-adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    reject = adjusted < 0.05
    return adjusted, reject


def run_effect_sizes():
    with open("verified_ablation_raw.json") as f:
        raw = json.load(f)

    rows = []
    pvals = []
    for cfg_a, cfg_b in COMPARISONS:
        for n in SIG_CHECKPOINTS:
            a = raw[cfg_a][str(n)]
            b = raw[cfg_b][str(n)]
            t_stat, p = stats.ttest_rel(a, b)
            d = cohens_d_paired(a, b)
            rows.append({
                "comparison": f"{cfg_a} vs {cfg_b}", "n": n,
                "mean_diff": float(np.mean(a) - np.mean(b)),
                "cohens_d": d, "t_stat": float(t_stat), "p_raw": float(p),
            })
            pvals.append(float(p))

    pvals = np.array(pvals)
    adjusted, reject = holm_bonferroni(pvals)
    for row, adj, rej in zip(rows, adjusted, reject):
        row["p_holm"] = float(adj)
        row["survives_holm_0.05"] = bool(rej)

    df = pd.DataFrame(rows)
    df.to_csv("statistical_rigor_effect_sizes.csv", index=False)
    print("\n=== EFFECT SIZES + HOLM-BONFERRONI CORRECTION (15 tests) ===")
    print(df.to_string(index=False))
    n_survive = int(reject.sum())
    print(f"\n{n_survive}/15 comparisons survive Holm-Bonferroni correction at alpha=0.05")
    return df


# ---------------------------------------------------------------------------
# Stage 2: pretraining-seed variation (Base TL vs AQT-alone, no fine-tuning)
# ---------------------------------------------------------------------------
SEED_GRID = [42, 123, 456, 789, 2024]
SEED_CHECKPOINT = "statistical_rigor_seeds_checkpoint.json"


def load_seed_checkpoint():
    try:
        with open(SEED_CHECKPOINT) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_seed_checkpoint(d):
    with open(SEED_CHECKPOINT, "w") as f:
        json.dump(d, f)


def run_seed_variation(time_budget_seconds=None):
    X17, y17, X18, y18, feature_cols = load_data()
    input_dim = X17.shape[1]
    done = load_seed_checkpoint()
    max_n = max(SAMPLE_CHECKPOINTS)
    t0 = time.time()

    for seed in SEED_GRID:
        if str(seed) in done:
            continue
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = pretrain_source_model(X17, y17, input_dim)
        pretrained_state = model.state_dict()
        source_pred_buffer = build_source_pred_buffer(model, X17, seed=seed)

        seed_results = {"1_base_tl": {n: [] for n in SAMPLE_CHECKPOINTS},
                         "3_tl_aqt": {n: [] for n in SAMPLE_CHECKPOINTS}}
        for mc in range(N_MONTE_CARLO):
            rng = np.random.RandomState(seed * 1000 + mc)
            X_stream, y_stream = make_cold_start_stream(X18, y18, max_n, 0.01, rng)
            for cfg_id, cfg in [("1_base_tl", BASE_CFG), ("3_tl_aqt", AQT_CFG)]:
                preds_all, y_true = run_one_trial(cfg_id, cfg, pretrained_state, input_dim,
                                                   X_stream, y_stream, source_pred_buffer, max_n)
                for n in SAMPLE_CHECKPOINTS:
                    f1 = f1_score(y_true[:n], preds_all[:n], average="macro", zero_division=0)
                    seed_results[cfg_id][n].append(f1)

        done[str(seed)] = seed_results
        save_seed_checkpoint(done)
        elapsed = time.time() - t0
        print(f"  [seed variation] seed={seed} done ({elapsed:.1f}s elapsed this invocation)", flush=True)
        if time_budget_seconds is not None and elapsed >= time_budget_seconds:
            print(f"Time budget reached; {len(done)}/{len(SEED_GRID)} seeds done. Re-run to continue.")
            return done
    return done


def summarize_seed_variation(done):
    rows = []
    for seed in SEED_GRID:
        if str(seed) not in done:
            continue
        sr = done[str(seed)]
        base_20k = sr["1_base_tl"]["20000"] if "20000" in sr["1_base_tl"] else sr["1_base_tl"][20000]
        aqt_20k = sr["3_tl_aqt"]["20000"] if "20000" in sr["3_tl_aqt"] else sr["3_tl_aqt"][20000]
        t_stat, p = stats.ttest_rel(aqt_20k, base_20k)
        d = cohens_d_paired(aqt_20k, base_20k)
        rows.append({
            "seed": seed,
            "base_tl_F1@20000_mean": float(np.mean(base_20k)),
            "aqt_F1@20000_mean": float(np.mean(aqt_20k)),
            "diff": float(np.mean(aqt_20k) - np.mean(base_20k)),
            "cohens_d": d, "p": float(p),
        })
    df = pd.DataFrame(rows)
    df.to_csv("statistical_rigor_seed_variation.csv", index=False)
    print("\n=== PRETRAINING-SEED VARIATION (AQT-alone vs Base TL, F1@20000) ===")
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Stage 3: pool bootstrap (resample target-pool row indices, not just the
# malicious-rate stream construction)
# ---------------------------------------------------------------------------
N_BOOTSTRAP = 50
BOOTSTRAP_CHECKPOINT = "statistical_rigor_bootstrap_checkpoint.json"


def load_bootstrap_checkpoint():
    try:
        with open(BOOTSTRAP_CHECKPOINT) as f:
            d = json.load(f)
        return d["completed"], d["base_f1"], d["aqt_f1"]
    except FileNotFoundError:
        return 0, [], []


def save_bootstrap_checkpoint(completed, base_f1, aqt_f1):
    with open(BOOTSTRAP_CHECKPOINT, "w") as f:
        json.dump({"completed": completed, "base_f1": base_f1, "aqt_f1": aqt_f1}, f)


def run_bootstrap(time_budget_seconds=None):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    X17, y17, X18, y18, feature_cols = load_data()
    input_dim = X17.shape[1]
    model = pretrain_source_model(X17, y17, input_dim)
    pretrained_state = model.state_dict()
    source_pred_buffer = build_source_pred_buffer(model, X17)

    completed, base_f1, aqt_f1 = load_bootstrap_checkpoint()
    max_n = max(SAMPLE_CHECKPOINTS)
    t0 = time.time()
    b = completed
    while b < N_BOOTSTRAP:
        boot_rng = np.random.RandomState(90000 + b)
        boot_idx = boot_rng.choice(len(X18), len(X18), replace=True)  # resample the POOL itself
        X18_boot, y18_boot = X18[boot_idx], y18[boot_idx]

        stream_rng = np.random.RandomState(SEED + b)
        X_stream, y_stream = make_cold_start_stream(X18_boot, y18_boot, max_n, 0.01, stream_rng)

        preds_base, y_true = run_one_trial("1_base_tl", BASE_CFG, pretrained_state, input_dim,
                                            X_stream, y_stream, source_pred_buffer, max_n)
        preds_aqt, _ = run_one_trial("3_tl_aqt", AQT_CFG, pretrained_state, input_dim,
                                      X_stream, y_stream, source_pred_buffer, max_n)
        base_f1.append(f1_score(y_true, preds_base, average="macro", zero_division=0))
        aqt_f1.append(f1_score(y_true, preds_aqt, average="macro", zero_division=0))

        b += 1
        save_bootstrap_checkpoint(b, base_f1, aqt_f1)
        elapsed = time.time() - t0
        if b % 10 == 0:
            print(f"  [bootstrap] {b}/{N_BOOTSTRAP} done ({elapsed:.1f}s elapsed this invocation)", flush=True)
        if time_budget_seconds is not None and elapsed >= time_budget_seconds:
            print(f"Time budget reached; {b}/{N_BOOTSTRAP} bootstrap reps done. Re-run to continue.")
            return b, base_f1, aqt_f1
    return b, base_f1, aqt_f1


def summarize_bootstrap(base_f1, aqt_f1):
    base_f1, aqt_f1 = np.array(base_f1), np.array(aqt_f1)
    diff = aqt_f1 - base_f1
    ci_lo, ci_hi = np.percentile(diff, [2.5, 97.5])
    print(f"\n=== POOL BOOTSTRAP ({len(diff)} reps, F1@20000, AQT-alone minus Base TL) ===")
    print(f"  mean diff: {diff.mean():.4f}, 95% percentile CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  diff > 0 in {int((diff > 0).sum())}/{len(diff)} bootstrap reps")
    with open("statistical_rigor_bootstrap.json", "w") as f:
        json.dump({
            "base_f1": base_f1.tolist(), "aqt_f1": aqt_f1.tolist(),
            "diff_mean": float(diff.mean()), "ci_95_lo": float(ci_lo), "ci_95_hi": float(ci_hi),
            "n_positive": int((diff > 0).sum()), "n_reps": len(diff),
        }, f, indent=2)


if __name__ == "__main__":
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else None

    if stage in ("all", "effect_sizes"):
        run_effect_sizes()
    if stage in ("all", "seeds"):
        done = run_seed_variation(time_budget_seconds=budget)
        if len(done) == len(SEED_GRID):
            summarize_seed_variation(done)
    if stage in ("all", "bootstrap"):
        completed, base_f1, aqt_f1 = run_bootstrap(time_budget_seconds=budget)
        if completed >= N_BOOTSTRAP:
            summarize_bootstrap(base_f1, aqt_f1)
