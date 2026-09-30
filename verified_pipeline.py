"""
Verified ATL Pipeline for ICC 2027 Revision
=============================================

This is a from-scratch, verified pipeline built directly against the real
data in this repository (data/robust_2017_final.csv, data/robust_2018_final.csv:
10 named, pre-selected flow features, NOT a 14-dim PCA projection).

It replaces the untraceable "Table II" harness. Every number this script
produces is reproducible by re-running it; nothing here is asserted without
being computed by this code.

Honest methodology notes (read before citing any output in the paper):
  - Feature space: 10 features, selected via RF importance / PCA-loading
    ranking (NOT a PCA projection). See PIPELINE_NOTES.md for detail.
  - Source domain: robust_2017_final.csv (100,000 rows, 19.8% attack)
  - Target domain: robust_2018_final.csv (100,000 rows, 16.8% attack)
  - Cold-start stream: the target corpus is subsampled (with replacement,
    matching the paper's Monte Carlo methodology) to a stated malicious
    rate (default 1%) to simulate a low-prevalence deployment stream.
    This IS a real resampling step, implemented here, not assumed.
  - AQT: per-sample FIFO buffer, W=500, pre-seeded with W source-domain
    predictions, threshold = quantile(buffer, q). q is a script parameter
    (set to 0.99 by default to match the paper's Section III-B1 value;
    change AQT_Q below and re-run to check sensitivity).
  - Algorithm 2: two-phase discriminative unfreezing exactly as coded in
    discriminative_lr_schedule.py (Phase 1 n<=1000 output-only, Phase 2
    sigmoid-scaled hidden-layer rate), reused here.
  - Config 5a "Full ATL (supervised, oracle upper bound)" fine-tunes on
    TRUE target labels -- this is NOT the proposed method, it is a ceiling
    for comparison, and must be labeled as such in the paper (see Reviewer
    A's core objection).
  - Config 5b "Full ATL (pseudo-label, proposed)" is Option A: fine-tunes
    only on confidence-gated pseudo-labels derived from AQT's own
    threshold. This is the actual proposed zero-label method.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import json
import time

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "data"
AQT_Q = 0.99
AQT_WINDOW = 500
PHASE1_CUTOFF = 1000
SIGMOID_K = 0.001
SIGMOID_X0 = 3600.0
BASE_LR = 0.00138
PRETRAIN_EPOCHS = 15
PRETRAIN_BATCH = 64
TARGET_MALICIOUS_RATE = 0.01     # the "1% stream" the paper describes
SAMPLE_CHECKPOINTS = [100, 1000, 5000, 10000, 20000]
N_MONTE_CARLO = 30                # matches the paper's stated Monte Carlo methodology
SEED = 42

# --- Fix 1: de-duplicate rows before any modeling or resampling ---
# robust_2018_final.csv is only 15.1% unique rows (95.3% of attack rows and
# 82.8% of benign rows are exact duplicates). Sampling with replacement from
# a corpus this duplicated repeatedly redraws the same handful of distinct
# points, which is what froze the AQT threshold in the first pass.
DEDUPLICATE = True

# --- Fix 2: percentile-based pseudo-label gating instead of a fixed
# absolute margin around tau_dyn. The target-domain probability distribution
# is compressed and skewed, so a fixed +/-0.05 band around tau starves
# positive pseudo-labels (2 of 5000 steps in the first pass). Gating by
# rank within the buffer instead adapts to whatever the distribution's
# actual shape is.
PSEUDO_HI_QUANTILE = 0.995   # stricter than AQT's own q=0.99 decision boundary
PSEUDO_LO_QUANTILE = 0.50    # below the buffer median counts as confident-benign

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Model (matches discriminative_lr_schedule.py / the code Kenny confirmed)
# ---------------------------------------------------------------------------
class CompactMLP(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.block1 = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU())
        self.block2 = nn.Sequential(nn.Linear(32, 16), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x):
        return self.out(self.block2(self.block1(x)))

    def output_params(self):
        return list(self.out.parameters())

    def hidden_params(self):
        return list(self.block1.parameters()) + list(self.block2.parameters())


def sigmoid_weight(n_t, k=SIGMOID_K, x0=SIGMOID_X0):
    return 1.0 / (1.0 + np.exp(-k * (n_t - x0)))


def get_aqt_threshold(buffer, q=AQT_Q, default_tau=0.5, window=AQT_WINDOW):
    if len(buffer) < window:
        return default_tau
    return float(np.quantile(list(buffer)[-window:], q))


def get_pseudo_bounds(buffer, window=AQT_WINDOW):
    """
    Percentile-based pseudo-label bounds (Fix 2). Returns (tau_hi, tau_lo):
    a prediction above tau_hi is a confident-positive pseudo-label, below
    tau_lo is a confident-negative pseudo-label, anything between is
    skipped as ambiguous. Both bounds are computed from the buffer's own
    distribution, not a fixed absolute offset from AQT's decision threshold.
    """
    if len(buffer) < window:
        return None, None
    recent = list(buffer)[-window:]
    tau_hi = float(np.quantile(recent, PSEUDO_HI_QUANTILE))
    tau_lo = float(np.quantile(recent, PSEUDO_LO_QUANTILE))
    return tau_hi, tau_lo


# ---------------------------------------------------------------------------
# Data loading + pretraining
# ---------------------------------------------------------------------------
def load_data():
    df17 = pd.read_csv(f"{DATA_DIR}/robust_2017_final.csv")
    df18 = pd.read_csv(f"{DATA_DIR}/robust_2018_final.csv")

    if DEDUPLICATE:
        n17_before, n18_before = len(df17), len(df18)
        df17 = df17.drop_duplicates().reset_index(drop=True)
        df18 = df18.drop_duplicates().reset_index(drop=True)
        print(f"  [dedup] 2017: {n17_before} -> {len(df17)} rows "
              f"({100*len(df17)/n17_before:.1f}% kept)")
        print(f"  [dedup] 2018: {n18_before} -> {len(df18)} rows "
              f"({100*len(df18)/n18_before:.1f}% kept)")

    feature_cols = [c for c in df17.columns if c != "Label"]

    scaler = StandardScaler().fit(df17[feature_cols].values)
    X17 = scaler.transform(df17[feature_cols].values)
    y17 = df17["Label"].values.astype(np.float32)
    X18 = scaler.transform(df18[feature_cols].values)
    y18 = df18["Label"].values.astype(np.float32)
    return X17, y17, X18, y18, feature_cols


def pretrain_source_model(X17, y17, input_dim, epochs=PRETRAIN_EPOCHS, batch_size=PRETRAIN_BATCH):
    model = CompactMLP(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    X = torch.tensor(X17, dtype=torch.float32)
    y = torch.tensor(y17, dtype=torch.float32).unsqueeze(1)

    model.train()
    n = X.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X[idx], y[idx]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        if epoch == epochs - 1:
            print(f"  [pretrain] final epoch loss: {total_loss / n:.4f}")
    return model


def build_source_pred_buffer(model, X17, window=AQT_WINDOW, seed=SEED):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X17), window, replace=False)
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X17[idx], dtype=torch.float32)).numpy().flatten()
    return preds.tolist()


def make_cold_start_stream(X18, y18, n_needed, malicious_rate, rng):
    """
    Resample the real target corpus (16.8% attack) down to the stated
    malicious_rate, WITH replacement, matching the paper's stated Monte
    Carlo resampling methodology. This is a real, implemented resampling
    step, not an assumed one.
    """
    benign_idx = np.where(y18 == 0)[0]
    attack_idx = np.where(y18 == 1)[0]

    n_attack = max(1, int(round(n_needed * malicious_rate)))
    n_benign = n_needed - n_attack

    chosen_benign = rng.choice(benign_idx, n_benign, replace=True)
    chosen_attack = rng.choice(attack_idx, n_attack, replace=True)

    chosen = np.concatenate([chosen_benign, chosen_attack])
    rng.shuffle(chosen)
    return X18[chosen], y18[chosen]


# ---------------------------------------------------------------------------
# Ablation configs
# ---------------------------------------------------------------------------
CONFIGS = {
    "1_base_tl":            dict(use_aqt=False, adapt=False, pseudo_label=False),
    "3_tl_aqt":              dict(use_aqt=True,  adapt=False, pseudo_label=False),
    "4_tl_unfreeze":         dict(use_aqt=False, adapt=True,  pseudo_label=False),
    "5a_full_atl_oracle":    dict(use_aqt=True,  adapt=True,  pseudo_label=False),
    "5b_full_atl_pseudo":    dict(use_aqt=True,  adapt=True,  pseudo_label=True),
}


def run_one_trial(config_id, cfg, pretrained_state, input_dim,
                   X_stream, y_stream, source_pred_buffer, n_samples):
    model = CompactMLP(input_dim=input_dim)
    model.load_state_dict(pretrained_state)

    buffer = list(source_pred_buffer) if cfg["use_aqt"] else []
    pseudo_pos_pool, pseudo_neg_pool = [], []

    preds_all, y_true = [], []
    criterion = nn.BCELoss()

    for t in range(1, n_samples + 1):
        x_t = torch.tensor(X_stream[t - 1:t], dtype=torch.float32)
        y_t = float(y_stream[t - 1])
        y_true.append(y_t)

        model.eval()
        with torch.no_grad():
            p_t = model(x_t).item()

        if cfg["use_aqt"]:
            buffer.append(p_t)
            tau = get_aqt_threshold(buffer)
        else:
            tau = 0.5
        preds_all.append(1 if p_t > tau else 0)

        if cfg["adapt"]:
            # decide the training target for this step
            if cfg["pseudo_label"]:
                tau_hi, tau_lo = get_pseudo_bounds(buffer)
                if tau_hi is None:
                    label_for_step = None  # buffer not full yet, no calibrated bounds
                elif p_t > tau_hi:
                    label_for_step = 1.0
                elif p_t < tau_lo:
                    label_for_step = 0.0
                else:
                    label_for_step = None  # ambiguous, skip adaptation this step
            else:
                label_for_step = y_t  # oracle: true label (upper bound only)

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

    return preds_all, y_true


def run_ablation(X17, y17, X18, y18, input_dim, feature_cols):
    print("Pretraining source-domain model on 2017 data...")
    model = pretrain_source_model(X17, y17, input_dim)
    pretrained_state = model.state_dict()
    source_pred_buffer = build_source_pred_buffer(model, X17)

    results = {cfg_id: {n: [] for n in SAMPLE_CHECKPOINTS} for cfg_id in CONFIGS}
    max_n = max(SAMPLE_CHECKPOINTS)

    t0 = time.time()
    for mc in range(N_MONTE_CARLO):
        rng = np.random.RandomState(SEED + mc)
        X_stream, y_stream = make_cold_start_stream(X18, y18, max_n, TARGET_MALICIOUS_RATE, rng)

        for cfg_id, cfg in CONFIGS.items():
            preds_all, y_true = run_one_trial(
                cfg_id, cfg, pretrained_state, input_dim,
                X_stream, y_stream, source_pred_buffer, max_n
            )
            for n in SAMPLE_CHECKPOINTS:
                f1 = f1_score(y_true[:n], preds_all[:n], average="macro", zero_division=0)
                results[cfg_id][n].append(f1)
        print(f"  MC run {mc+1}/{N_MONTE_CARLO} done ({time.time()-t0:.1f}s elapsed)")

    return results


def summarize(results):
    rows = []
    for cfg_id in CONFIGS:
        row = {"config": cfg_id}
        for n in SAMPLE_CHECKPOINTS:
            vals = results[cfg_id][n]
            row[f"F1@{n}_mean"] = float(np.mean(vals))
            row[f"F1@{n}_std"] = float(np.std(vals))
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Loading real pipeline data (10-feature robust_2017/2018_final.csv)...")
    X17, y17, X18, y18, feature_cols = load_data()
    input_dim = X17.shape[1]
    print(f"  input_dim = {input_dim}, features = {feature_cols}")
    print(f"  source (2017): {len(X17)} rows, {100*y17.mean():.1f}% attack")
    print(f"  target (2018): {len(X18)} rows, {100*y18.mean():.1f}% attack")
    print(f"  simulating cold-start stream at {100*TARGET_MALICIOUS_RATE:.1f}% malicious rate")
    print(f"  AQT: W={AQT_WINDOW}, q={AQT_Q}  |  Monte Carlo runs: {N_MONTE_CARLO}")
    print()

    results = run_ablation(X17, y17, X18, y18, input_dim, feature_cols)
    summary_df = summarize(results)

    print("\n=== RESULTS (macro-F1, mean +/- std over", N_MONTE_CARLO, "runs) ===")
    print(summary_df.to_string(index=False))

    summary_df.to_csv("verified_ablation_results.csv", index=False)
    with open("verified_ablation_raw.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: verified_ablation_results.csv, verified_ablation_raw.json")
