"""
Infiltration Case Study - Verified Version
=============================================

Re-runs the cold-start evaluation on REAL Infiltration-labeled CSE-CIC-IDS2018
traffic (Wednesday-28-02-2018 + Thursday-01-03-2018 raw CICFlowMeter exports,
restricted to Benign vs Infilteration rows), instead of the original
submission's unreproducible Table V numbers.

Source-domain pretraining is unchanged (robust_2017_final.csv, same as
verified_pipeline.py). Target domain here is data/infiltration_2018.csv
(built by extracting the same 10 pipeline features from the two raw daily
CIC-IDS2018 files and keeping only Benign/Infilteration rows), NOT
robust_2018_final.csv (which mixes all attack types).

Honest methodology notes:
  - The raw daily files are, once reduced to these 10 features, 92% exact
    duplicate rows (75,241 of 944,171) -- consistent with, and further
    confirming, the Leevy & Khoshgoftaar duplicate-row artifact already
    disclosed in Section IV-A / Limitations. Deduplication is applied
    before any resampling, exactly as for the main pipeline.
  - Cold-start window: first 7,200 samples (matching the original
    manuscript's Table V window), at a 1% malicious (Infiltration) rate,
    30 Monte Carlo runs.
  - Reports per-class macro-F1 for THIS binary problem (Benign vs
    Infiltration only) -- comparable in spirit to the original Table V
    "Infiltr." row, not a re-derivation of the other five attack
    categories (DoS, DDoS, Web Attack, Brute Force, Botnet), which are
    not attempted here.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import json
import time

from verified_pipeline import (
    CompactMLP, AQT_WINDOW, SEED, N_MONTE_CARLO,
    pretrain_source_model, build_source_pred_buffer,
    run_one_trial, DATA_DIR,
)

TARGET_MALICIOUS_RATE = 0.01
COLDSTART_WINDOW = 7200
INFIL_FILE = f"{DATA_DIR}/infiltration_2018.csv"

CONFIGS = {
    "1_base_tl":         dict(use_aqt=False, adapt=False, pseudo_label=False),
    "3_tl_aqt":          dict(use_aqt=True,  adapt=False, pseudo_label=False),
    "5a_full_atl_oracle": dict(use_aqt=True,  adapt=True,  pseudo_label=False),
    "5b_full_atl_pseudo": dict(use_aqt=True,  adapt=True,  pseudo_label=True),
}

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_data():
    df17 = pd.read_csv(f"{DATA_DIR}/robust_2017_final.csv").drop_duplicates().reset_index(drop=True)
    dfi = pd.read_csv(INFIL_FILE)

    feature_cols = [c for c in df17.columns if c != "Label"]
    # infiltration_2018.csv was already built with matching column names/order intent;
    # re-select explicitly to guarantee identical column order to the source scaler.
    assert set(feature_cols).issubset(set(dfi.columns)), "feature mismatch vs source pipeline"

    scaler = StandardScaler().fit(df17[feature_cols].values)
    X17 = scaler.transform(df17[feature_cols].values)
    y17 = df17["Label"].values.astype(np.float32)
    Xi = scaler.transform(dfi[feature_cols].values)
    yi = dfi["Label"].values.astype(np.float32)
    return X17, y17, Xi, yi, feature_cols


def make_infiltration_stream(Xi, yi, n_needed, malicious_rate, rng):
    benign_idx = np.where(yi == 0)[0]
    attack_idx = np.where(yi == 1)[0]
    n_attack = max(1, int(round(n_needed * malicious_rate)))
    n_benign = n_needed - n_attack
    chosen_benign = rng.choice(benign_idx, n_benign, replace=True)
    chosen_attack = rng.choice(attack_idx, n_attack, replace=True)
    chosen = np.concatenate([chosen_benign, chosen_attack])
    rng.shuffle(chosen)
    return Xi[chosen], yi[chosen]


def main():
    print("Loading source (2017, deduped) + infiltration-only target (Wed 28/02 + Thu 01/03, deduped)...")
    X17, y17, Xi, yi, feature_cols = load_data()
    input_dim = X17.shape[1]
    print(f"  input_dim = {input_dim}")
    print(f"  source (2017): {len(X17)} rows, {100*y17.mean():.1f}% attack")
    print(f"  target (infiltration-only, deduped): {len(Xi)} rows, {100*yi.mean():.1f}% infiltration")
    print(f"  cold-start window: {COLDSTART_WINDOW} samples @ {100*TARGET_MALICIOUS_RATE:.1f}% malicious, {N_MONTE_CARLO} MC runs")

    print("\nPretraining source-domain model on 2017 data (identical to verified_pipeline.py)...")
    model = pretrain_source_model(X17, y17, input_dim)
    pretrained_state = model.state_dict()
    source_pred_buffer = build_source_pred_buffer(model, X17)

    results = {cfg_id: [] for cfg_id in CONFIGS}
    t0 = time.time()
    for mc in range(N_MONTE_CARLO):
        rng = np.random.RandomState(SEED + mc)
        X_stream, y_stream = make_infiltration_stream(Xi, yi, COLDSTART_WINDOW, TARGET_MALICIOUS_RATE, rng)
        for cfg_id, cfg in CONFIGS.items():
            preds_all, y_true = run_one_trial(
                cfg_id, cfg, pretrained_state, input_dim,
                X_stream, y_stream, source_pred_buffer, COLDSTART_WINDOW
            )
            f1 = f1_score(y_true, preds_all, average="macro", zero_division=0)
            results[cfg_id].append(f1)
        print(f"  MC run {mc+1}/{N_MONTE_CARLO} done ({time.time()-t0:.1f}s elapsed)")

    print("\n=== INFILTRATION CASE STUDY RESULTS (macro-F1, mean +/- std, 30 runs, n=7200 window) ===")
    summary = {}
    for cfg_id in CONFIGS:
        vals = results[cfg_id]
        summary[cfg_id] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        print(f"  {cfg_id:24s}  F1 = {np.mean(vals):.3f} +/- {np.std(vals):.3f}")

    with open("infiltration_case_study_raw.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("infiltration_case_study_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved: infiltration_case_study_raw.json, infiltration_case_study_results.json")


if __name__ == "__main__":
    main()
