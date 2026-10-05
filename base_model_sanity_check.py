"""
Base-Model Sanity Check (Round 5, Reviewer 1 major concern 1)
==================================================================

Reviewer 1: "Static AUROC is 0.5852 with 10 features and 0.5848 with 77,
on different raw day files. Agreement to the third decimal is very unlikely
by chance. Confirm the rerun really feeds 77 inputs to the MLP (about 3,041
parameters). ... Candidates are column misalignment or unit and direction
differences between the two CICFlowMeter releases. Report in-domain source
AUROC, per-family cross-domain AUROC, and a per-feature shift check."

This script answers each item directly, on both representations:

  A. Architecture: parameter count and first-layer shape for input_dim=77
     and input_dim=10 (3,041 and 897 expected).
  B. In-domain AUROC: pretrain on 80% of source, evaluate on held-out 20%
     of source, for both representations.
  C. Static cross-domain AUROC across 5 pretraining seeds for both
     representations (is 0.5852 vs 0.5848 a coincidence or structural?).
  D. Per-family cross-domain AUROC (77-feature raw files only: the 10-feature
     pipeline CSVs carry no family labels).
  E. Column-alignment and direction check (77-feature): per-feature
     univariate AUROC in source vs target (sign agreement, rank correlation),
     and a "diagonal" test: does each target column's benign distribution
     match the SAME-index source column better than any other source column
     (low KS) -- if columns were misaligned this would fail.
  F. Per-feature shift: target mean/std in source-sigma units.

Data: the same raw-file cleaning/subsampling as all_features_rerun.py
(same RandomState(SEED) subsample), but keeping the string family labels.
"""

import json
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from verified_pipeline import SEED, CompactMLP, pretrain_source_model, load_data
from all_features_rerun import SRC_2017, TGT_2018, DROP_2017, DROP_2018, POOL_SIZE, load_raw

SEEDS = [42, 123, 456, 789, 2024]


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy().flatten()


def predict_logit(model, X):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32)
        return model.out[0](model.block2(model.block1(x))).numpy().flatten()


def load_77_with_families():
    df17 = load_raw(SRC_2017, DROP_2017)
    df18 = load_raw(TGT_2018, DROP_2018)
    feats = [c for c in df17.columns if c != "Label"]
    feats18 = [c for c in df18.columns if c != "Label"]
    assert len(feats) == len(feats18) == 77

    def prep(df, cols):
        fam = df["Label"].astype(str).str.strip()
        y = (fam.str.upper() != "BENIGN").astype(np.float32)
        X = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        out = pd.concat([X, y.rename("Label"), fam.rename("Family")], axis=1).dropna()
        out = out.drop_duplicates(subset=list(cols) + ["Label"]).reset_index(drop=True)
        return out

    d17, d18 = prep(df17, feats), prep(df18, feats18)
    rng = np.random.RandomState(SEED)
    if len(d17) > POOL_SIZE:
        d17 = d17.iloc[rng.choice(len(d17), POOL_SIZE, replace=False)].reset_index(drop=True)
    if len(d18) > POOL_SIZE:
        d18 = d18.iloc[rng.choice(len(d18), POOL_SIZE, replace=False)].reset_index(drop=True)
    scaler = StandardScaler().fit(d17[feats].values)
    return (scaler.transform(d17[feats].values), d17["Label"].values.astype(np.float32), d17["Family"].values,
            scaler.transform(d18[feats18].values), d18["Label"].values.astype(np.float32), d18["Family"].values,
            feats, feats18)


def main():
    out = {}

    # ---- A. architecture
    print("=== A. Architecture ===")
    for d in (77, 10):
        m = CompactMLP(input_dim=d)
        n = sum(p.numel() for p in m.parameters())
        print(f"  input_dim={d}: {n} trainable parameters, first-layer weight shape {tuple(m.block1[0].weight.shape)}")
        out[f"params_{d}"] = int(n)

    # ---- load both representations
    print("\nLoading 10-feature pipeline data and 77-feature raw-file data...")
    X17_10, y17_10, X18_10, y18_10, cols10 = load_data()
    X17_77, y17_77, fam17, X18_77, y18_77, fam18, feats17, feats18 = load_77_with_families()
    print(f"  10-feature: source {X17_10.shape}, target {X18_10.shape}")
    print(f"  77-feature: source {X17_77.shape}, target {X18_77.shape}")
    print("  77-feature target families:", dict(pd.Series(fam18).value_counts()))
    print("  77-feature source families:", dict(pd.Series(fam17).value_counts()))

    # ---- B. in-domain AUROC (80/20 source split)
    print("\n=== B. In-domain source AUROC (train 80% / test 20% of source) ===")
    out["in_domain_auroc"] = {}
    for name, X, y in (("10-feature", X17_10, y17_10), ("77-feature", X17_77, y17_77)):
        aucs = []
        for sd in SEEDS:
            torch.manual_seed(sd)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=sd, stratify=y)
            m = pretrain_source_model(Xtr, ytr, X.shape[1])
            aucs.append(float(roc_auc_score(yte, predict(m, Xte))))
        print(f"  {name}: in-domain AUROC {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}  (seeds {SEEDS}: {[round(a,4) for a in aucs]})")
        out["in_domain_auroc"][name] = aucs

    # ---- C. static cross-domain AUROC across seeds (probability AND logit space)
    print("\n=== C. Static cross-domain AUROC across 5 pretraining seeds (train on ALL source) ===")
    print("  prob = AUROC on sigmoid outputs (what the paper reports); logit = AUROC on pre-sigmoid scores;")
    print("  sat1/sat0 = fraction of target scores that are float-exactly 1.0 / 0.0 (ties).")
    out["cross_domain_auroc"] = {}
    models77 = {}
    for name, X17, y17, X18, y18 in (("10-feature", X17_10, y17_10, X18_10, y18_10),
                                      ("77-feature", X17_77, y17_77, X18_77, y18_77)):
        rows = []
        for sd in SEEDS:
            torch.manual_seed(sd)
            m = pretrain_source_model(X17, y17, X17.shape[1])
            pr = predict(m, X18); lg = predict_logit(m, X18)
            rows.append({"seed": sd, "auroc_prob": float(roc_auc_score(y18, pr)),
                         "auroc_logit": float(roc_auc_score(y18, lg)),
                         "sat1": float((pr == 1.0).mean()), "sat0": float((pr == 0.0).mean())})
            if name == "77-feature":
                models77[sd] = m
        ap = [r["auroc_prob"] for r in rows]; al = [r["auroc_logit"] for r in rows]
        print(f"  {name}: prob  {np.mean(ap):.4f} +/- {np.std(ap):.4f}  {[round(a,4) for a in ap]}")
        print(f"  {name}: logit {np.mean(al):.4f} +/- {np.std(al):.4f}  {[round(a,4) for a in al]}")
        print(f"  {name}: sat1 {[round(r['sat1'],3) for r in rows]}  sat0 {[round(r['sat0'],3) for r in rows]}")
        out["cross_domain_auroc"][name] = rows

    # ---- D. per-family cross-domain AUROC (77-feature model, each seed; prob and logit)
    print("\n=== D. Per-family cross-domain AUROC (77-feature model) ===")
    fam18s = pd.Series(fam18).astype(str).values
    benign = (pd.Series(fam18s).str.upper() == "BENIGN").values
    out["per_family_cross_domain"] = {}
    for sd in SEEDS:
        m = models77[sd]
        pr = predict(m, X18_77); lg = predict_logit(m, X18_77)
        for fam in sorted(set(fam18s[~benign])):
            mask = benign | (fam18s == fam)
            n_f = int((fam18s == fam).sum())
            yb = (fam18s[mask] == fam).astype(int)
            ap = float(roc_auc_score(yb, pr[mask])); al = float(roc_auc_score(yb, lg[mask]))
            print(f"  seed {sd:5d}  {fam:24s} n={n_f:6d}  AUROC prob {ap:.4f}  logit {al:.4f}")
            out["per_family_cross_domain"].setdefault(fam, {})[str(sd)] = {"n": n_f, "prob": ap, "logit": al}
    # within-source (in-sample) per-family, seed 42
    fam17s = pd.Series(fam17).astype(str).values
    benign17 = (pd.Series(fam17s).str.upper() == "BENIGN").values
    s17 = predict_logit(models77[42], X17_77)
    out["per_family_source_in_sample"] = {}
    for fam in sorted(set(fam17s[~benign17])):
        mask = benign17 | (fam17s == fam)
        n_f = int((fam17s == fam).sum())
        if n_f < 20:
            continue
        auc = float(roc_auc_score((fam17s[mask] == fam).astype(int), s17[mask]))
        print(f"  [source, in-sample, seed 42, logit] {fam:20s} n={n_f:6d}  AUROC {auc:.4f}")
        out["per_family_source_in_sample"][fam] = {"n": n_f, "auroc_logit": auc}

    # ---- E. column alignment / direction check (77-feature)
    print("\n=== E. Column-alignment and direction check (77-feature) ===")
    rng = np.random.RandomState(0)
    b17 = X17_77[y17_77 == 0]; b18 = X18_77[y18_77 == 0]
    b17s = b17[rng.choice(len(b17), 3000, replace=False)]
    b18s = b18[rng.choice(len(b18), 3000, replace=False)]
    ks = np.zeros((77, 77))
    for i in range(77):
        for j in range(77):
            ks[i, j] = ks_2samp(b18s[:, j], b17s[:, i]).statistic  # target col j vs source col i
    diag = np.diag(ks)
    best_src = ks.argmin(axis=0)
    n_diag_best = int((best_src == np.arange(77)).sum())
    n_diag_top5 = int(sum(j in np.argsort(ks[:, j])[:5] for j in range(77)))
    print(f"  Diagonal KS (same-index columns): mean {diag.mean():.3f}; off-diagonal mean {ks[~np.eye(77, dtype=bool)].mean():.3f}")
    print(f"  Target column j best matches source column j (min KS) for {n_diag_best}/77 features; within top-5 for {n_diag_top5}/77")
    out["alignment"] = {"diag_ks_mean": float(diag.mean()),
                         "offdiag_ks_mean": float(ks[~np.eye(77, dtype=bool)].mean()),
                         "diag_is_best": n_diag_best, "diag_in_top5": n_diag_top5}

    def uni_auc(X, y):
        return np.array([roc_auc_score(y, X[:, j]) for j in range(X.shape[1])])
    ua17 = uni_auc(X17_77, y17_77)
    ua18 = uni_auc(X18_77, y18_77)
    rho = spearmanr(ua17 - 0.5, ua18 - 0.5).correlation
    same_sign = int(((ua17 - 0.5) * (ua18 - 0.5) > 0).sum())
    strong = (np.abs(ua17 - 0.5) > 0.1) & (np.abs(ua18 - 0.5) > 0.1)
    flipped_strong = int((((ua17 - 0.5) * (ua18 - 0.5) < 0) & strong).sum())
    print(f"  Univariate (attack vs benign) AUROC, source vs target: Spearman rho={rho:.3f}; "
          f"same direction for {same_sign}/77 features; strongly-informative-in-both but FLIPPED: {flipped_strong}")
    out["direction"] = {"spearman": float(rho), "same_sign": same_sign, "flipped_strong": flipped_strong}
    flipped_names = [feats18[j] for j in range(77) if strong[j] and (ua17[j]-0.5)*(ua18[j]-0.5) < 0]
    print("  flipped features (|AUROC-0.5|>0.1 in both):", flipped_names)
    out["direction"]["flipped_names"] = flipped_names

    # ---- F. per-feature shift
    print("\n=== F. Per-feature shift (target in source-sigma units; benign-only) ===")
    mu = b18.mean(axis=0); sd = b18.std(axis=0)
    shift = pd.DataFrame({"feature": feats18, "target_mean_in_src_sigma": mu, "target_std_in_src_sigma": sd,
                          "ks_same_index": diag, "uni_auroc_source": ua17, "uni_auroc_target": ua18})
    shift["abs_mean_shift"] = shift["target_mean_in_src_sigma"].abs()
    shift = shift.sort_values("abs_mean_shift", ascending=False)
    print(shift.head(10).to_string(index=False))
    print(f"  features with |benign mean shift| > 1 sigma: {(shift['abs_mean_shift'] > 1).sum()}/77; > 0.5 sigma: {(shift['abs_mean_shift'] > 0.5).sum()}/77")
    shift.to_csv("base_model_sanity_check_feature_shift.csv", index=False)

    # the paper's 10 pipeline features
    print("\n  Paper's 10 pipeline features in the 10-feature data (univariate AUROC, source vs target):")
    ua17_10, ua18_10 = uni_auc(X17_10, y17_10), uni_auc(X18_10, y18_10)
    for c, a, b in zip(cols10, ua17_10, ua18_10):
        print(f"    {c:28s} source {a:.3f}  target {b:.3f}")
    out["pipeline10_univariate"] = {c: [float(a), float(b)] for c, a, b in zip(cols10, ua17_10, ua18_10)}

    with open("base_model_sanity_check_raw.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: base_model_sanity_check_raw.json, base_model_sanity_check_feature_shift.csv")


if __name__ == "__main__":
    main()
