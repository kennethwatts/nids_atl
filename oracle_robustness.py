"""
Robustness check: is the oracle (true-label) fine-tuning baseline (config 5a)
a fair upper bound, or a strawman that collapses only because of an
unreasonably aggressive, unweighted learning rate?

Tests three variants of oracle fine-tuning, all using true target labels
(same as 5a), keeping everything else (AQT, discriminative unfreezing
schedule, dedup, stream construction, seeds) identical to the main pipeline
so results are directly paired and comparable to verified_ablation_raw.json:

  5a_oracle_lowlr          : same class-weighting as original (none), but
                              base LR reduced 10x (BASE_LR * 0.1)
  5a_oracle_classweighted  : original LR, but BCE loss weighted 99:1 in
                              favor of the positive (attack) class, matching
                              the stream's ~1% malicious rate
  5a_oracle_lowlr_cw       : both fixes combined

Re-uses load_data / pretrain_source_model / build_source_pred_buffer /
make_cold_start_stream / CompactMLP / sigmoid_weight / get_aqt_threshold
from verified_pipeline.py unchanged, so the only thing that differs from
config 5a is the learning rate scale and/or loss weighting in the
adaptation step.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import json
import time

from verified_pipeline import (
    load_data, pretrain_source_model, build_source_pred_buffer,
    make_cold_start_stream, CompactMLP, sigmoid_weight, get_aqt_threshold,
    PHASE1_CUTOFF, BASE_LR, SEED, SAMPLE_CHECKPOINTS, N_MONTE_CARLO,
    TARGET_MALICIOUS_RATE,
)

ROBUST_CONFIGS = {
    "5a_oracle_lowlr":         dict(lr_scale=0.1, pos_weight=1.0),
    "5a_oracle_classweighted": dict(lr_scale=1.0, pos_weight=99.0),
    "5a_oracle_lowlr_cw":      dict(lr_scale=0.1, pos_weight=99.0),
}


def run_oracle_variant(cfg, pretrained_state, input_dim, X_stream, y_stream, source_pred_buffer, n_samples):
    model = CompactMLP(input_dim=input_dim)
    model.load_state_dict(pretrained_state)

    buffer = list(source_pred_buffer)  # always uses AQT, matching config 5a
    preds_all, y_true = [], []
    lr_scale = cfg["lr_scale"]
    pos_weight = cfg["pos_weight"]

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

        # oracle: always adapt on the true label
        model.train()
        omega = sigmoid_weight(t)
        lr = BASE_LR * lr_scale
        if t <= PHASE1_CUTOFF:
            optimizer = torch.optim.Adam(model.output_params(), lr=lr)
        else:
            optimizer = torch.optim.Adam([
                {"params": model.output_params(), "lr": lr},
                {"params": model.hidden_params(), "lr": 0.1 * lr * omega},
            ])
        w = pos_weight if y_t == 1.0 else 1.0
        criterion = nn.BCELoss(weight=torch.tensor([[w]], dtype=torch.float32))
        optimizer.zero_grad()
        target = torch.tensor([[y_t]], dtype=torch.float32)
        loss = criterion(model(x_t), target)
        loss.backward()
        optimizer.step()

    return preds_all, y_true


def main():
    print("Loading data...")
    X17, y17, X18, y18, feature_cols = load_data()
    input_dim = X17.shape[1]

    print("Pretraining source model...")
    model = pretrain_source_model(X17, y17, input_dim)
    pretrained_state = model.state_dict()
    source_pred_buffer = build_source_pred_buffer(model, X17)

    results = {cfg_id: {n: [] for n in SAMPLE_CHECKPOINTS} for cfg_id in ROBUST_CONFIGS}
    max_n = max(SAMPLE_CHECKPOINTS)

    t0 = time.time()
    for mc in range(N_MONTE_CARLO):
        rng = np.random.RandomState(SEED + mc)  # identical stream construction to the main pipeline
        X_stream, y_stream = make_cold_start_stream(X18, y18, max_n, TARGET_MALICIOUS_RATE, rng)

        for cfg_id, cfg in ROBUST_CONFIGS.items():
            preds_all, y_true = run_oracle_variant(
                cfg, pretrained_state, input_dim, X_stream, y_stream, source_pred_buffer, max_n
            )
            for n in SAMPLE_CHECKPOINTS:
                f1 = f1_score(y_true[:n], preds_all[:n], average="macro", zero_division=0)
                results[cfg_id][n].append(f1)
        print(f"  MC run {mc+1}/{N_MONTE_CARLO} done ({time.time()-t0:.1f}s elapsed)")

    rows = []
    for cfg_id in ROBUST_CONFIGS:
        row = {"config": cfg_id}
        for n in SAMPLE_CHECKPOINTS:
            vals = results[cfg_id][n]
            row[f"F1@{n}_mean"] = float(np.mean(vals))
            row[f"F1@{n}_std"] = float(np.std(vals))
        rows.append(row)
    summary_df = pd.DataFrame(rows)

    print("\n=== ROBUSTNESS RESULTS (macro-F1, mean +/- std over", N_MONTE_CARLO, "runs) ===")
    print(summary_df.to_string(index=False))

    summary_df.to_csv("oracle_robustness_results.csv", index=False)
    with open("oracle_robustness_raw.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: oracle_robustness_results.csv, oracle_robustness_raw.json")


if __name__ == "__main__":
    main()
