"""
Snapshot-Model PR-AUC + Dedup Sensitivity Run
==================================================

Round 4 (Oct 4, 2026) Reviewer 2 item #5 and Reviewer 1 item #4.

PART A -- SNAPSHOT PR-AUC (Reviewer 2 item #5):
"The ranking claim is still confounded. PR-AUC is pooled over a
drifting model... Use snapshot models on a fixed held-out set."

Section IV-C.1's PR-AUC (Table III) pools raw scores across the WHOLE
20,000-sample non-stationary stream for each configuration. For Base TL
and AQT-alone this is harmless -- neither ever changes the model, so
"pooled over a drifting model" does not apply to them (AQT only moves
the decision threshold; their identical PR-AUC is a direct, correct
consequence of using the identical frozen scores, not a confound). It
DOES apply to the two fine-tuning configs (oracle, pseudo-label), whose
model weights change at every adaptation step, so pooling mixes score
distributions from many different model-states together.

This script freezes a FIXED held-out evaluation set (drawn once, at the
paper's 1% malicious rate, independent of the cold-start streams) and,
for the oracle and pseudo-label configs, takes a SNAPSHOT of the model
at each of the paper's checkpoints (n=100/1000/5000/10000/20000) during
each of 30 Monte Carlo trials, evaluating that frozen snapshot's PR-AUC
on the fixed held-out set. This replaces "PR-AUC pooled over a moving
target" with "PR-AUC of the model AS IT STOOD at checkpoint n, measured
on a set that never changes." Base TL and AQT-alone are evaluated once
(their score is identical at every checkpoint by construction, so no
snapshotting is needed, and this script says so rather than computing a
trivial per-checkpoint loop for them).

PART B -- DEDUP SENSITIVITY (Reviewer 1 item #4):
"Dedup remains an active filter, not just a caveat. It drops 95.3% of
attack rows, so the attack pool is a small, atypical remnant. Add a
sensitivity run with raw-record or no dedup, and report pool sizes and
per-family composition."

Reruns Base TL and AQT-alone (no fine-tuning needed; isolates AQT's own
behavior) with DEDUPLICATE=False, reporting pool sizes before/after and
the resulting F1 comparison. Per-family composition (DoS/DDoS/Web
Attack/Brute Force/Botnet/etc.) CANNOT be reported here: robust_2018_
final.csv carries only a binary Label column, and the only family-
labeled raw data available in this environment is the two Infiltration-
specific CIC-IDS2018 day files used for Section IV-D (Benign vs.
Infiltration only, by construction -- there is no DoS/DDoS/Web Attack/
Brute Force/Botnet traffic in those two files to break out). This is a
genuine data-availability gap, not an oversight, and is reported as such
rather than fabricated.
"""

import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, average_precision_score

from verified_pipeline import (
    SEED, N_MONTE_CARLO, SAMPLE_CHECKPOINTS, AQT_WINDOW, PHASE1_CUTOFF, BASE_LR,
    CONFIGS, CompactMLP, sigmoid_weight, get_aqt_threshold, get_pseudo_bounds,
    DATA_DIR, make_cold_start_stream, pretrain_source_model, build_source_pred_buffer,
    run_one_trial,
)
from sklearn.preprocessing import StandardScaler

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Part A: snapshot PR-AUC
# ---------------------------------------------------------------------------
def run_one_trial_with_snapshots(cfg, pretrained_state, input_dim, X_stream, y_stream,
                                  source_pred_buffer, n_samples, checkpoints, held_X, held_y):
    model = CompactMLP(input_dim=input_dim)
    model.load_state_dict(pretrained_state)
    buffer = list(source_pred_buffer) if cfg["use_aqt"] else []
    criterion = nn.BCELoss()
    checkpoints_set = set(checkpoints)
    snapshot_prauc = {}
    held_X_t = torch.tensor(held_X, dtype=torch.float32)

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

        if t in checkpoints_set:
            model.eval()
            with torch.no_grad():
                held_scores = model(held_X_t).numpy().flatten()
            snapshot_prauc[t] = float(average_precision_score(held_y, held_scores))

    return snapshot_prauc


SNAPSHOT_CHECKPOINT = "snapshot_prauc_checkpoint.json"


def load_snapshot_checkpoint():
    try:
        with open(SNAPSHOT_CHECKPOINT) as f:
            ck = json.load(f)
        results = {cfg_id: {int(n): v for n, v in by_n.items()} for cfg_id, by_n in ck["results"].items()}
        return ck["done"], results
    except FileNotFoundError:
        return {}, {"5a_full_atl_oracle": {n: [] for n in SAMPLE_CHECKPOINTS},
                     "5b_full_atl_pseudo": {n: [] for n in SAMPLE_CHECKPOINTS}}


def save_snapshot_checkpoint(done, results):
    with open(SNAPSHOT_CHECKPOINT, "w") as f:
        json.dump({"done": done, "results": results}, f)


def run_snapshot_prauc(pretrained_state, input_dim, X18, y18, source_pred_buffer,
                        held_X, held_y, time_budget_seconds=None):
    done, results = load_snapshot_checkpoint()
    max_n = max(SAMPLE_CHECKPOINTS)
    t0 = time.time()

    for cfg_id in ["5a_full_atl_oracle", "5b_full_atl_pseudo"]:
        mc_done = done.get(cfg_id, 0)
        if mc_done >= N_MONTE_CARLO:
            continue
        cfg = CONFIGS[cfg_id]
        mc = mc_done
        while mc < N_MONTE_CARLO:
            rng = np.random.RandomState(SEED + mc)
            X_stream, y_stream = make_cold_start_stream(X18, y18, max_n, 0.01, rng)
            snap = run_one_trial_with_snapshots(cfg, pretrained_state, input_dim, X_stream, y_stream,
                                                 source_pred_buffer, max_n, SAMPLE_CHECKPOINTS, held_X, held_y)
            for n in SAMPLE_CHECKPOINTS:
                results[cfg_id][n].append(snap[n])
            mc += 1
            done[cfg_id] = mc
            save_snapshot_checkpoint(done, results)
            elapsed = time.time() - t0
            print(f"  [snapshot PR-AUC] {cfg_id} MC {mc}/{N_MONTE_CARLO} done ({elapsed:.1f}s)", flush=True)
            if time_budget_seconds is not None and elapsed >= time_budget_seconds:
                print("Time budget reached. Re-run to continue.")
                return done, results
    return done, results


# ---------------------------------------------------------------------------
# Part B: dedup sensitivity
# ---------------------------------------------------------------------------
def load_data_no_dedup():
    df17 = pd.read_csv(f"{DATA_DIR}/robust_2017_final.csv")
    df18 = pd.read_csv(f"{DATA_DIR}/robust_2018_final.csv")
    feature_cols = [c for c in df17.columns if c != "Label"]
    scaler = StandardScaler().fit(df17[feature_cols].values)
    X17 = scaler.transform(df17[feature_cols].values)
    y17 = df17["Label"].values.astype(np.float32)
    X18 = scaler.transform(df18[feature_cols].values)
    y18 = df18["Label"].values.astype(np.float32)
    pool_info = {
        "n18_rows": len(df18), "n18_attack_rows": int(y18.sum()),
        "n18_attack_pct": float(100 * y18.mean()),
    }
    return X17, y17, X18, y18, feature_cols, pool_info


def run_dedup_sensitivity(pretrained_state, input_dim, source_pred_buffer):
    X17_nd, y17_nd, X18_nd, y18_nd, feature_cols, pool_info = load_data_no_dedup()
    # Also need the deduplicated pool sizes for comparison (recomputed here for self-containment)
    df17_dd = pd.read_csv(f"{DATA_DIR}/robust_2017_final.csv").drop_duplicates()
    df18_dd = pd.read_csv(f"{DATA_DIR}/robust_2018_final.csv").drop_duplicates()
    dedup_info = {
        "n18_rows_dedup": len(df18_dd), "n18_attack_rows_dedup": int(df18_dd["Label"].sum()),
        "n18_attack_pct_dedup": float(100 * df18_dd["Label"].mean()),
    }

    results = {"1_base_tl": {n: [] for n in SAMPLE_CHECKPOINTS},
               "3_tl_aqt": {n: [] for n in SAMPLE_CHECKPOINTS}}
    max_n = max(SAMPLE_CHECKPOINTS)
    t0 = time.time()
    for mc in range(N_MONTE_CARLO):
        rng = np.random.RandomState(SEED + mc)
        X_stream, y_stream = make_cold_start_stream(X18_nd, y18_nd, max_n, 0.01, rng)
        for cfg_id in ["1_base_tl", "3_tl_aqt"]:
            preds_all, y_true = run_one_trial(cfg_id, CONFIGS[cfg_id], pretrained_state, input_dim,
                                               X_stream, y_stream, source_pred_buffer, max_n)
            for n in SAMPLE_CHECKPOINTS:
                f1 = f1_score(y_true[:n], preds_all[:n], average="macro", zero_division=0)
                results[cfg_id][n].append(f1)
        if (mc + 1) % 10 == 0:
            print(f"  [dedup sensitivity] MC run {mc+1}/{N_MONTE_CARLO} done ({time.time()-t0:.1f}s elapsed)")

    return results, pool_info, dedup_info


def main(stage="all", time_budget_seconds=None):
    print("Loading data, pretraining source model (identical to verified_pipeline.py, dedup ON)...")
    from verified_pipeline import load_data
    X17, y17, X18, y18, feature_cols = load_data()
    input_dim = X17.shape[1]
    model = pretrain_source_model(X17, y17, input_dim)
    pretrained_state = model.state_dict()
    source_pred_buffer = build_source_pred_buffer(model, X17)

    if stage in ("all", "snapshot"):
        print("\n--- Part A: snapshot PR-AUC on a fixed held-out set ---")
        held_rng = np.random.RandomState(99999)
        X_held, y_held = make_cold_start_stream(X18, y18, 5000, 0.01, held_rng)
        with torch.no_grad():
            frozen_scores = model(torch.tensor(X_held, dtype=torch.float32)).numpy().flatten()
        frozen_prauc = float(average_precision_score(y_held, frozen_scores))
        print(f"  Base TL / AQT-alone PR-AUC on fixed held-out set (frozen model, constant at every "
              f"checkpoint by construction): {frozen_prauc:.4f}")

        done, snap_results = run_snapshot_prauc(pretrained_state, input_dim, X18, y18, source_pred_buffer,
                                                  X_held, y_held, time_budget_seconds=time_budget_seconds)
        if all(done.get(c, 0) >= N_MONTE_CARLO for c in ["5a_full_atl_oracle", "5b_full_atl_pseudo"]):
            rows = [{"config": "1_base_tl / 3_tl_aqt (frozen, no snapshotting needed)",
                     **{f"PR-AUC@{n}": frozen_prauc for n in SAMPLE_CHECKPOINTS}}]
            for cfg_id in ["5a_full_atl_oracle", "5b_full_atl_pseudo"]:
                row = {"config": cfg_id}
                for n in SAMPLE_CHECKPOINTS:
                    vals = snap_results[cfg_id][n]
                    row[f"PR-AUC@{n}"] = float(np.mean(vals))
                    row[f"PR-AUC@{n}_std"] = float(np.std(vals))
                rows.append(row)
            df = pd.DataFrame(rows)
            df.to_csv("snapshot_prauc_results.csv", index=False)
            print("\n=== SNAPSHOT PR-AUC (fixed held-out set, 30 MC runs) ===")
            print(df.to_string(index=False))
            with open("snapshot_prauc_raw.json", "w") as f:
                json.dump({"frozen_prauc": frozen_prauc, "snapshot_results": snap_results}, f, indent=2)
            print("Saved: snapshot_prauc_results.csv, snapshot_prauc_raw.json")

    if stage in ("all", "dedup"):
        print("\n--- Part B: dedup sensitivity (Base TL + AQT-alone, no dedup) ---")
        results, pool_info, dedup_info = run_dedup_sensitivity(pretrained_state, input_dim, source_pred_buffer)
        rows = []
        for cfg_id in results:
            row = {"config": cfg_id}
            for n in SAMPLE_CHECKPOINTS:
                vals = results[cfg_id][n]
                row[f"F1@{n}_mean"] = float(np.mean(vals))
                row[f"F1@{n}_std"] = float(np.std(vals))
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv("dedup_sensitivity_results.csv", index=False)
        print("\n=== DEDUP SENSITIVITY (no dedup, macro-F1, 30 MC runs) ===")
        print(f"  Pool sizes: no-dedup 2018 rows={pool_info['n18_rows']}, "
              f"attack rows={pool_info['n18_attack_rows']} ({pool_info['n18_attack_pct']:.2f}%)")
        print(f"             deduped 2018 rows={dedup_info['n18_rows_dedup']}, "
              f"attack rows={dedup_info['n18_attack_rows_dedup']} ({dedup_info['n18_attack_pct_dedup']:.2f}%)")
        print(df.to_string(index=False))
        print("\n  NOTE: per-family attack composition (DoS/DDoS/Web Attack/Brute Force/Botnet/etc.) "
              "cannot be reported -- robust_2018_final.csv carries only a binary Label column, and the "
              "only family-labeled raw data available in this environment (the two Infiltration-specific "
              "CIC-IDS2018 day files) contains Benign/Infiltration traffic only, by construction. This is "
              "a genuine data-availability gap, to be disclosed as such rather than fabricated.")
        with open("dedup_sensitivity_raw.json", "w") as f:
            json.dump({"pool_info": pool_info, "dedup_info": dedup_info, "results": results}, f, indent=2)
        print("Saved: dedup_sensitivity_results.csv, dedup_sensitivity_raw.json")


if __name__ == "__main__":
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else None
    main(stage=stage, time_budget_seconds=budget)
