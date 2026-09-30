"""
Precision/Recall/PR-AUC Analysis - Verified Version
=======================================================

Addresses the pre-submission review's core finding: macro-F1 alone
obscures what is actually happening at the attack-class level under 1%
imbalance (a trivial always-benign classifier already scores macro-F1
~0.4975). This script reports attack-class precision, recall, F1, and
PR-AUC (average precision on raw scores) directly, for the same configs,
checkpoints, seeds, and pretrained model as verified_pipeline.py and
infiltration_case_study.py -- nothing here is a new experiment, only
additional metrics computed from the same runs.

Also reports the frozen source-pretrained model's static AUROC on the
full deduplicated target corpus (not a cold-start stream), to check
whether cross-domain transfer is producing a meaningfully informative
score distribution at all.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score, precision_score, recall_score, average_precision_score, roc_auc_score
)
import json
import time

from verified_pipeline import (
    CompactMLP, AQT_WINDOW, SEED, N_MONTE_CARLO, SAMPLE_CHECKPOINTS, DATA_DIR,
    load_data, pretrain_source_model, build_source_pred_buffer,
    make_cold_start_stream, TARGET_MALICIOUS_RATE,
    get_aqt_threshold, get_pseudo_bounds, sigmoid_weight,
    PHASE1_CUTOFF, BASE_LR,
)

CONFIGS = {
    "1_base_tl":          dict(use_aqt=False, adapt=False, pseudo_label=False),
    "3_tl_aqt":           dict(use_aqt=True,  adapt=False, pseudo_label=False),
    "5a_full_atl_oracle": dict(use_aqt=True,  adapt=True,  pseudo_label=False),
    "5b_full_atl_pseudo": dict(use_aqt=True,  adapt=True,  pseudo_label=True),
}

torch.manual_seed(SEED)
np.random.seed(SEED)


def run_one_trial_scored(cfg, pretrained_state, input_dim,
                          X_stream, y_stream, source_pred_buffer, n_samples):
    """Identical to verified_pipeline.run_one_trial, but also returns the
    raw probability score p_t for every step (needed for PR-AUC)."""
    model = CompactMLP(input_dim=input_dim)
    model.load_state_dict(pretrained_state)

    buffer = list(source_pred_buffer) if cfg["use_aqt"] else []
    preds_all, scores_all, y_true = [], [], []
    criterion = nn.BCELoss()

    for t in range(1, n_samples + 1):
        x_t = torch.tensor(X_stream[t - 1:t], dtype=torch.float32)
        y_t = float(y_stream[t - 1])
        y_true.append(y_t)

        model.eval()
        with torch.no_grad():
            p_t = model(x_t).item()
        scores_all.append(p_t)

        if cfg["use_aqt"]:
            buffer.append(p_t)
            tau = get_aqt_threshold(buffer)
        else:
            tau = 0.5
        preds_all.append(1 if p_t > tau else 0)

        if cfg["adapt"]:
            if cfg["pseudo_label"]:
                tau_hi, tau_lo = get_pseudo_bounds(buffer)
                if tau_hi is None:
                    label_for_step = None
                elif p_t > tau_hi:
                    label_for_step = 1.0
                elif p_t < tau_lo:
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
                loss = criterion(model(x_t), target)
                loss.backward()
                optimizer.step()

    return preds_all, scores_all, y_true


def analyze(X17, y17, X, y, stream_fn, checkpoints, n_mc, max_n, malicious_rate, tag):
    input_dim = X17.shape[1]
    print(f"[{tag}] pretraining source model...")
    model = pretrain_source_model(X17, y17, input_dim)
    pretrained_state = model.state_dict()
    source_pred_buffer = build_source_pred_buffer(model, X17)

    # static frozen-model AUROC on the full deduplicated target corpus
    model.eval()
    with torch.no_grad():
        full_scores = model(torch.tensor(X, dtype=torch.float32)).numpy().flatten()
    static_auroc = roc_auc_score(y, full_scores)
    print(f"[{tag}] frozen model static target-domain AUROC (full corpus): {static_auroc:.4f}")

    results = {cfg_id: {n: {"precision": [], "recall": [], "attack_f1": [], "pr_auc": []}
                         for n in checkpoints} for cfg_id in CONFIGS}

    t0 = time.time()
    for mc in range(n_mc):
        rng = np.random.RandomState(SEED + mc)
        X_stream, y_stream = stream_fn(X, y, max_n, malicious_rate, rng)
        for cfg_id, cfg in CONFIGS.items():
            preds_all, scores_all, y_true = run_one_trial_scored(
                cfg, pretrained_state, input_dim, X_stream, y_stream, source_pred_buffer, max_n
            )
            preds_all = np.array(preds_all); scores_all = np.array(scores_all); y_true = np.array(y_true)
            for n in checkpoints:
                yt, pr, sc = y_true[:n], preds_all[:n], scores_all[:n]
                results[cfg_id][n]["precision"].append(precision_score(yt, pr, zero_division=0))
                results[cfg_id][n]["recall"].append(recall_score(yt, pr, zero_division=0))
                results[cfg_id][n]["attack_f1"].append(f1_score(yt, pr, pos_label=1, zero_division=0))
                # PR-AUC needs both classes present to be meaningful
                if len(np.unique(yt)) > 1:
                    results[cfg_id][n]["pr_auc"].append(average_precision_score(yt, sc))
        print(f"  [{tag}] MC run {mc+1}/{n_mc} done ({time.time()-t0:.1f}s elapsed)")

    return results, static_auroc


def summarize(results, checkpoints, tag):
    rows = []
    for cfg_id in CONFIGS:
        row = {"config": cfg_id}
        for n in checkpoints:
            for metric in ["precision", "recall", "attack_f1", "pr_auc"]:
                vals = results[cfg_id][n][metric]
                row[f"{metric}@{n}_mean"] = float(np.mean(vals)) if vals else float("nan")
                row[f"{metric}@{n}_std"] = float(np.std(vals)) if vals else float("nan")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"precision_recall_{tag}_results.csv", index=False)
    print(f"\n=== {tag.upper()} (attack-class precision/recall/F1/PR-AUC) ===")
    for n in checkpoints:
        print(f"-- n={n} --")
        for cfg_id in CONFIGS:
            p = np.mean(results[cfg_id][n]["precision"])
            r = np.mean(results[cfg_id][n]["recall"])
            f = np.mean(results[cfg_id][n]["attack_f1"])
            auc_vals = results[cfg_id][n]["pr_auc"]
            auc = np.mean(auc_vals) if auc_vals else float("nan")
            print(f"  {cfg_id:24s} P={p:.4f} R={r:.4f} attackF1={f:.4f} PR-AUC={auc:.4f}")
    return df


if __name__ == "__main__":
    print("=== Main cold-start ablation ===")
    X17, y17, X18, y18, feature_cols = load_data()
    main_results, main_auroc = analyze(
        X17, y17, X18, y18, make_cold_start_stream, SAMPLE_CHECKPOINTS,
        N_MONTE_CARLO, max(SAMPLE_CHECKPOINTS), TARGET_MALICIOUS_RATE, "main"
    )
    main_df = summarize(main_results, SAMPLE_CHECKPOINTS, "main")
    with open("precision_recall_main_raw.json", "w") as f:
        json.dump(main_results, f, indent=2)

    print("\n=== Infiltration case study ===")
    from infiltration_case_study import load_data as load_infil_data, make_infiltration_stream, COLDSTART_WINDOW
    X17b, y17b, Xi, yi, _ = load_infil_data()
    infil_results, infil_auroc = analyze(
        X17b, y17b, Xi, yi, make_infiltration_stream, [COLDSTART_WINDOW],
        N_MONTE_CARLO, COLDSTART_WINDOW, TARGET_MALICIOUS_RATE, "infiltration"
    )
    infil_df = summarize(infil_results, [COLDSTART_WINDOW], "infiltration")
    with open("precision_recall_infiltration_raw.json", "w") as f:
        json.dump(infil_results, f, indent=2)

    with open("static_auroc_results.json", "w") as f:
        json.dump({"main_target_auroc": main_auroc, "infiltration_target_auroc": infil_auroc}, f, indent=2)

    print("\nSaved: precision_recall_main_results.csv, precision_recall_infiltration_results.csv, static_auroc_results.json")
