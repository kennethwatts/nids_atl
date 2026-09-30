"""
AQT Quantile Sensitivity Sweep - Verified Version
=====================================================

Resolves Reviewer A's AQT-quantile conflict (original manuscript cited
q=0.99 in Section III-B1 but q=0.9010 from an Optuna search in Section
IV-A, with no explanation of which number actually produced the headline
results) by directly measuring cold-start F1 for the "TL + AQT, no
fine-tune" configuration across the grid search values named in the
original text: q in {0.80, 0.85, 0.90, 0.95, 0.99}.

Uses the same pretrained source model, source-prediction buffer, and
cold-start stream construction as verified_pipeline.py -- only AQT_Q
varies across runs. No fine-tuning is applied (adapt=False), isolating
the quantile choice's effect on AQT's own contribution specifically.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
import json
import time

from verified_pipeline import (
    SEED, N_MONTE_CARLO, SAMPLE_CHECKPOINTS, DATA_DIR,
    load_data, pretrain_source_model, build_source_pred_buffer,
    make_cold_start_stream, run_one_trial, TARGET_MALICIOUS_RATE,
)
import verified_pipeline as vp

Q_GRID = [0.80, 0.85, 0.90, 0.95, 0.99]
CFG = dict(use_aqt=True, adapt=False, pseudo_label=False)  # "3_tl_aqt", no fine-tune


def make_threshold_fn(q):
    """
    Note: get_aqt_threshold's default arg `q=AQT_Q` is bound at module
    load time, so simply setting vp.AQT_Q does NOT change the threshold
    run_one_trial computes (it calls get_aqt_threshold(buffer) with no
    explicit q). Instead we replace the function object itself; since
    run_one_trial looks up `get_aqt_threshold` in the module's global
    namespace at call time (not at def time), this correctly changes
    the threshold used on every subsequent call.
    """
    def f(buffer, window=vp.AQT_WINDOW):
        if len(buffer) < window:
            return 0.5
        return float(np.quantile(list(buffer)[-window:], q))
    return f

torch.manual_seed(SEED)
np.random.seed(SEED)


def main():
    print("Loading data, pretraining source model (identical to verified_pipeline.py)...")
    X17, y17, X18, y18, feature_cols = load_data()
    input_dim = X17.shape[1]
    model = pretrain_source_model(X17, y17, input_dim)
    pretrained_state = model.state_dict()
    source_pred_buffer = build_source_pred_buffer(model, X17)

    results = {q: {n: [] for n in SAMPLE_CHECKPOINTS} for q in Q_GRID}
    max_n = max(SAMPLE_CHECKPOINTS)
    t0 = time.time()

    for q in Q_GRID:
        vp.get_aqt_threshold = make_threshold_fn(q)
        for mc in range(N_MONTE_CARLO):
            rng = np.random.RandomState(SEED + mc)
            X_stream, y_stream = make_cold_start_stream(X18, y18, max_n, TARGET_MALICIOUS_RATE, rng)
            preds_all, y_true = run_one_trial(
                "3_tl_aqt", CFG, pretrained_state, input_dim,
                X_stream, y_stream, source_pred_buffer, max_n
            )
            for n in SAMPLE_CHECKPOINTS:
                f1 = f1_score(y_true[:n], preds_all[:n], average="macro", zero_division=0)
                results[q][n].append(f1)
        print(f"  q={q} done ({time.time()-t0:.1f}s elapsed)")

    print("\n=== AQT QUANTILE SENSITIVITY (macro-F1, mean +/- std, 30 MC runs) ===")
    rows = []
    for q in Q_GRID:
        row = {"q": q}
        for n in SAMPLE_CHECKPOINTS:
            vals = results[q][n]
            row[f"F1@{n}_mean"] = float(np.mean(vals))
            row[f"F1@{n}_std"] = float(np.std(vals))
        rows.append(row)
        print(f"  q={q}: " + ", ".join(f"F1@{n}={row[f'F1@{n}_mean']:.3f}" for n in SAMPLE_CHECKPOINTS))

    pd.DataFrame(rows).to_csv("aqt_quantile_sensitivity_results.csv", index=False)
    with open("aqt_quantile_sensitivity_raw.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: aqt_quantile_sensitivity_results.csv, aqt_quantile_sensitivity_raw.json")


if __name__ == "__main__":
    main()
