"""
Diagnostic: why does oracle (true-label) fine-tuning (config 5a) degrade
relative to AQT-alone (config 3) and baseline (config 1)?

Instruments a single, representative cold-start stream (same construction
as verified_pipeline.py, same seed) and records, every 50 steps:
  - mean raw model probability on true-benign samples seen so far in a
    rolling window (last 500 steps)
  - mean raw model probability on true-attack samples seen so far in the
    same rolling window
  - the AQT threshold tau at that step (for AQT-using configs)
  - the fraction of predictions that are positive in the rolling window
  - the model's final Linear(16,1) output-layer bias term (a single-number
    proxy for "is the decision function drifting toward always-negative")

Averaged over 10 Monte Carlo streams (not all 30, to keep this fast) for
each of: 1_base_tl, 3_tl_aqt, 5a_full_atl_oracle, 5b_full_atl_pseudo.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json

from verified_pipeline import (
    load_data, pretrain_source_model, build_source_pred_buffer,
    make_cold_start_stream, CompactMLP, sigmoid_weight, get_aqt_threshold,
    get_pseudo_bounds, PHASE1_CUTOFF, BASE_LR, SEED, AQT_WINDOW,
    TARGET_MALICIOUS_RATE,
)

RECORD_EVERY = 50
N_STEPS = 20000
N_DIAG_RUNS = 10  # fewer than the full 30 for speed; this is a diagnostic, not a final number

CONFIGS = {
    "1_base_tl":         dict(use_aqt=False, adapt=False, pseudo_label=False),
    "3_tl_aqt":          dict(use_aqt=True,  adapt=False, pseudo_label=False),
    "5a_full_atl_oracle": dict(use_aqt=True,  adapt=True,  pseudo_label=False),
    "5b_full_atl_pseudo": dict(use_aqt=True,  adapt=True,  pseudo_label=True),
}


def instrumented_trial(cfg, pretrained_state, input_dim, X_stream, y_stream, source_pred_buffer, n_samples):
    model = CompactMLP(input_dim=input_dim)
    model.load_state_dict(pretrained_state)

    buffer = list(source_pred_buffer) if cfg["use_aqt"] else []
    criterion = nn.BCELoss()

    recent_p_benign, recent_p_attack, recent_preds = [], [], []
    record = {"step": [], "mean_p_benign": [], "mean_p_attack": [], "tau": [],
              "frac_pred_positive": [], "output_bias": [], "n_adapt_steps_so_far": []}
    n_adapt = 0

    for t in range(1, n_samples + 1):
        x_t = torch.tensor(X_stream[t - 1:t], dtype=torch.float32)
        y_t = float(y_stream[t - 1])

        model.eval()
        with torch.no_grad():
            p_t = model(x_t).item()

        if cfg["use_aqt"]:
            buffer.append(p_t)
            tau = get_aqt_threshold(buffer)
        else:
            tau = 0.5
        pred_t = 1 if p_t > tau else 0

        (recent_p_attack if y_t == 1.0 else recent_p_benign).append(p_t)
        recent_preds.append(pred_t)
        if len(recent_p_benign) > 500:
            recent_p_benign.pop(0)
        if len(recent_p_attack) > 500:
            recent_p_attack.pop(0)
        if len(recent_preds) > 500:
            recent_preds.pop(0)

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
                n_adapt += 1
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

        if t % RECORD_EVERY == 0:
            out_bias = model.out[0].bias.item()
            record["step"].append(t)
            record["mean_p_benign"].append(float(np.mean(recent_p_benign)) if recent_p_benign else np.nan)
            record["mean_p_attack"].append(float(np.mean(recent_p_attack)) if recent_p_attack else np.nan)
            record["tau"].append(tau)
            record["frac_pred_positive"].append(float(np.mean(recent_preds)))
            record["output_bias"].append(out_bias)
            record["n_adapt_steps_so_far"].append(n_adapt)

    return record


def main():
    print("Loading data...")
    X17, y17, X18, y18, feature_cols = load_data()
    input_dim = X17.shape[1]

    print("Pretraining source model...")
    model = pretrain_source_model(X17, y17, input_dim)
    pretrained_state = model.state_dict()
    source_pred_buffer = build_source_pred_buffer(model, X17)

    all_records = {cfg_id: [] for cfg_id in CONFIGS}

    for mc in range(N_DIAG_RUNS):
        rng = np.random.RandomState(SEED + mc)
        X_stream, y_stream = make_cold_start_stream(X18, y18, N_STEPS, TARGET_MALICIOUS_RATE, rng)
        for cfg_id, cfg in CONFIGS.items():
            rec = instrumented_trial(cfg, pretrained_state, input_dim, X_stream, y_stream, source_pred_buffer, N_STEPS)
            all_records[cfg_id].append(rec)
        print(f"  MC diag run {mc+1}/{N_DIAG_RUNS} done")

    # Average across MC runs at each recorded step (steps are aligned since RECORD_EVERY is fixed)
    averaged = {}
    for cfg_id, recs in all_records.items():
        steps = recs[0]["step"]
        avg = {"step": steps}
        for key in ["mean_p_benign", "mean_p_attack", "tau", "frac_pred_positive", "output_bias", "n_adapt_steps_so_far"]:
            arr = np.array([r[key] for r in recs], dtype=float)  # (N_DIAG_RUNS, n_recorded_steps)
            avg[key + "_mean"] = np.nanmean(arr, axis=0).tolist()
            avg[key + "_std"] = np.nanstd(arr, axis=0).tolist()
        averaged[cfg_id] = avg

    with open("oracle_diagnostic.json", "w") as f:
        json.dump(averaged, f, indent=2)
    print("Saved oracle_diagnostic.json")


if __name__ == "__main__":
    main()
