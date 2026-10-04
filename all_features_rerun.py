"""
All-Feature (77-Feature) Core Ablation Rerun
=================================================

Round 4 (Oct 4, 2026) Reviewer 1 item #3, Meta-Assessment priority #6
(the heaviest item): "Everything is conditional on a near-chance base
model. Static AUROC is 0.585 and PR-AUC is 0.018 against 0.010 chance,
and the 10 PCA-loading features collide on 85% of rows... Rerun the
core ablation with all 78 features... to show whether the ordering
holds when the base signal is stronger."

DATA PROVENANCE: this environment had no record of which original raw
CICIDS2017/CSE-CIC-IDS2018 day files fed the existing 10-feature
robust_2017_final.csv / robust_2018_final.csv (no documentation
anywhere in the repo), so this is NOT a re-extraction of the same
underlying rows with more columns kept -- it is a fresh sample from
different raw day files that Kenny supplied:
  - Source (2017): Wednesday-workingHours.pcap_ISCX.csv (692,703 rows;
    DoS Hulk/GoldenEye/slowloris/Slowhttptest/Heartbleed vs. BENIGN)
  - Target (2018): Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv
    (1,048,575 rows; DDoS HOIC/LOIC-UDP vs. Benign)
Both files have much higher raw attack rates (~65%) and different
attack families than the paper's original 10-feature samples (DoS/DDoS
here vs. a broader mix there), so this is a DIFFERENT, not identical,
cold-start scenario. It answers the reviewer's actual question (does
the AQT-vs-fine-tuning ordering hold with a stronger, more-featured base
model) but is not a controlled like-for-like replacement for Table I,
and the paper must say so explicitly if these numbers are used.

FEATURE HARMONIZATION: the two files use different CICFlowMeter column-
naming conventions (2017 "ISCX" verbose names vs. 2018 abbreviated
names) and have small structural differences:
  - 2018 has 'Protocol' and 'Timestamp' columns 2017 lacks -- dropped.
  - 2017 has a duplicated 'Fwd Header Length' column (auto-renamed
    'Fwd Header Length.1' by pandas on read) that 2018 does not
    reproduce -- dropped.
After these two drops, the remaining 77 feature columns (+ Label) align
one-to-one by position/semantics between the two files (verified by
hand, column by column, before writing this script -- see FEATURE_MAP
below). This is 77, not exactly 78, common features; reported as such
rather than rounded up.

METHODOLOGY: mirrors verified_pipeline.py exactly except for the data
source and input_dim: inf/NaN rows dropped, exact-duplicate rows
dropped (now computed on the full 77-feature representation, not just
10 features), both domains randomly subsampled to 100,000 rows (fixed
seed) to stay comparable in scale to the paper's original samples
(composition will differ -- see provenance note above), StandardScaler
fit on source only, same CompactMLP architecture (now input_dim=77),
same CONFIGS, same SAMPLE_CHECKPOINTS, same 1% cold-start malicious
rate, same 30 Monte Carlo runs, same run_one_trial logic (imported
directly from verified_pipeline.py, unmodified).
"""

import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from verified_pipeline import (
    SEED, N_MONTE_CARLO, SAMPLE_CHECKPOINTS, CONFIGS,
    CompactMLP, pretrain_source_model, build_source_pred_buffer,
    make_cold_start_stream, run_one_trial,
)

torch.manual_seed(SEED)
np.random.seed(SEED)

SRC_2017 = "/mnt/user-data/uploads/Wednesday-workingHours_pcap_ISCX.csv"
TGT_2018 = "/mnt/user-data/uploads/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv"
POOL_SIZE = 100000

# 2017 columns to drop entirely (no 2018 counterpart)
DROP_2017 = ["Fwd Header Length.1"]
# 2018 columns to drop entirely (no 2017 counterpart)
DROP_2018 = ["Protocol", "Timestamp"]


def load_raw(path, drop_cols):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df


def clean_and_align(df17, df18):
    # After dropping DROP_2017/DROP_2018, both frames should have the same
    # number of columns (77 features + Label) in the same semantic order --
    # verified by hand against FEATURE_MAP logic in the module docstring.
    assert len(df17.columns) == len(df18.columns), \
        f"column count mismatch after drops: 2017={len(df17.columns)}, 2018={len(df18.columns)}"
    feature_cols_17 = [c for c in df17.columns if c != "Label"]
    feature_cols_18 = [c for c in df18.columns if c != "Label"]
    assert len(feature_cols_17) == len(feature_cols_18) == 77, \
        f"expected 77 common features, got {len(feature_cols_17)} (2017) / {len(feature_cols_18)} (2018)"

    def binarize_label(s):
        return (s.astype(str).str.strip().str.upper() != "BENIGN").astype(np.float32)

    y17 = binarize_label(df17["Label"])
    y18 = binarize_label(df18["Label"])

    X17 = df17[feature_cols_17].apply(pd.to_numeric, errors="coerce")
    X18 = df18[feature_cols_18].apply(pd.to_numeric, errors="coerce")
    X17 = X17.replace([np.inf, -np.inf], np.nan)
    X18 = X18.replace([np.inf, -np.inf], np.nan)

    df17_clean = pd.concat([X17, y17.rename("Label")], axis=1).dropna()
    df18_clean = pd.concat([X18, y18.rename("Label")], axis=1).dropna()

    n17_before_dedup, n18_before_dedup = len(df17_clean), len(df18_clean)
    df17_clean = df17_clean.drop_duplicates().reset_index(drop=True)
    df18_clean = df18_clean.drop_duplicates().reset_index(drop=True)

    pool_info = {
        "n17_raw": int(n17_before_dedup), "n17_deduped": int(len(df17_clean)),
        "n17_attack_pct_deduped": float(100 * df17_clean["Label"].mean()),
        "n18_raw": int(n18_before_dedup), "n18_deduped": int(len(df18_clean)),
        "n18_attack_pct_deduped": float(100 * df18_clean["Label"].mean()),
    }

    rng = np.random.RandomState(SEED)
    if len(df17_clean) > POOL_SIZE:
        idx = rng.choice(len(df17_clean), POOL_SIZE, replace=False)
        df17_clean = df17_clean.iloc[idx].reset_index(drop=True)
    if len(df18_clean) > POOL_SIZE:
        idx = rng.choice(len(df18_clean), POOL_SIZE, replace=False)
        df18_clean = df18_clean.iloc[idx].reset_index(drop=True)

    pool_info["n17_final"] = len(df17_clean)
    pool_info["n17_attack_pct_final"] = float(100 * df17_clean["Label"].mean())
    pool_info["n18_final"] = len(df18_clean)
    pool_info["n18_attack_pct_final"] = float(100 * df18_clean["Label"].mean())

    scaler = StandardScaler().fit(df17_clean[feature_cols_17].values)
    X17_scaled = scaler.transform(df17_clean[feature_cols_17].values)
    y17_arr = df17_clean["Label"].values.astype(np.float32)
    X18_scaled = scaler.transform(df18_clean[feature_cols_18].values)
    y18_arr = df18_clean["Label"].values.astype(np.float32)

    return X17_scaled, y17_arr, X18_scaled, y18_arr, feature_cols_17, pool_info


CHECKPOINT_PATH = "all_features_rerun_checkpoint.json"


def load_checkpoint():
    try:
        with open(CHECKPOINT_PATH) as f:
            ck = json.load(f)
        results = {cfg_id: {int(n): v for n, v in by_n.items()} for cfg_id, by_n in ck["results"].items()}
        return ck["completed_mc_runs"], results
    except FileNotFoundError:
        return 0, {cfg_id: {n: [] for n in SAMPLE_CHECKPOINTS} for cfg_id in CONFIGS}


def save_checkpoint(completed_mc_runs, results):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"completed_mc_runs": completed_mc_runs, "results": results}, f)


def main(time_budget_seconds=None):
    completed_mc_runs, results = load_checkpoint()

    print("Loading and harmonizing raw 78-feature CICFlowMeter files...")
    df17 = load_raw(SRC_2017, DROP_2017)
    df18 = load_raw(TGT_2018, DROP_2018)
    X17, y17, X18, y18, feature_cols, pool_info = clean_and_align(df17, df18)
    input_dim = X17.shape[1]
    print(f"  input_dim = {input_dim} features")
    print(f"  source (2017): raw-clean={pool_info['n17_raw']}, deduped={pool_info['n17_deduped']} "
          f"({pool_info['n17_attack_pct_deduped']:.1f}% attack), "
          f"final sample={pool_info['n17_final']} ({pool_info['n17_attack_pct_final']:.1f}% attack)")
    print(f"  target (2018): raw-clean={pool_info['n18_raw']}, deduped={pool_info['n18_deduped']} "
          f"({pool_info['n18_attack_pct_deduped']:.1f}% attack), "
          f"final sample={pool_info['n18_final']} ({pool_info['n18_attack_pct_final']:.1f}% attack)")

    print("\nPretraining source-domain model (77-feature CompactMLP)...")
    model = pretrain_source_model(X17, y17, input_dim)
    pretrained_state = model.state_dict()
    source_pred_buffer = build_source_pred_buffer(model, X17)

    with torch.no_grad():
        from sklearn.metrics import roc_auc_score
        static_scores = model(torch.tensor(X18, dtype=torch.float32)).numpy().flatten()
    static_auroc = float(roc_auc_score(y18, static_scores))
    print(f"  Static AUROC on full target pool (77 features): {static_auroc:.4f} "
          f"(paper's 10-feature figure: 0.5852)")

    max_n = max(SAMPLE_CHECKPOINTS)
    t0 = time.time()
    mc = completed_mc_runs
    while mc < N_MONTE_CARLO:
        rng = np.random.RandomState(SEED + mc)
        X_stream, y_stream = make_cold_start_stream(X18, y18, max_n, 0.01, rng)
        for cfg_id, cfg in CONFIGS.items():
            preds_all, y_true = run_one_trial(
                cfg_id, cfg, pretrained_state, input_dim,
                X_stream, y_stream, source_pred_buffer, max_n
            )
            for n in SAMPLE_CHECKPOINTS:
                f1 = f1_score(y_true[:n], preds_all[:n], average="macro", zero_division=0)
                results[cfg_id][n].append(f1)
        mc += 1
        save_checkpoint(mc, results)
        elapsed = time.time() - t0
        print(f"  MC run {mc}/{N_MONTE_CARLO} done ({elapsed:.1f}s elapsed this invocation)", flush=True)
        if time_budget_seconds is not None and elapsed >= time_budget_seconds:
            print(f"Time budget reached; checkpointed at {mc}/{N_MONTE_CARLO}. Re-run to continue.")
            return

    rows = []
    for cfg_id in CONFIGS:
        row = {"config": cfg_id}
        for n in SAMPLE_CHECKPOINTS:
            vals = results[cfg_id][n]
            row[f"F1@{n}_mean"] = float(np.mean(vals))
            row[f"F1@{n}_std"] = float(np.std(vals))
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"\n=== ALL-FEATURE (77-feature) ABLATION RESULTS (macro-F1, {N_MONTE_CARLO} MC runs) ===")
    print(df.to_string(index=False))

    df.to_csv("all_features_rerun_results.csv", index=False)
    with open("all_features_rerun_raw.json", "w") as f:
        json.dump({"pool_info": pool_info, "static_auroc": static_auroc,
                   "feature_cols": feature_cols, "results": results}, f, indent=2)
    print("\nSaved: all_features_rerun_results.csv, all_features_rerun_raw.json")


if __name__ == "__main__":
    import sys
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else None
    main(time_budget_seconds=budget)
