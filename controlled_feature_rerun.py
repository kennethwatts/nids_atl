"""
Controlled feature-set rerun on the SAME pool, in probability AND logit space
(Round 5 Reviewer 1 major concerns 1-2).

Why this exists: base_model_sanity_check.py showed the earlier 77-feature
rerun (all_features_rerun.py) ran AQT on sigmoid outputs of which 93.5% were
float-exactly 0.0 for the paper's pretraining seed, so its AUROC (0.5848) and
its AQT/oracle/pseudo "tie" were artifacts of score ties. Reviewer 1 also
asked for the feature set to be the ONLY thing that changes. Here the pool is
fixed (the 2017/2018 raw day files from raw_pool.py) and only the input
columns change: all 77 features vs the paper's ten pipeline features, mapped
into the raw columns by name.

Design: 5 pretraining seeds x 6 streams per seed = 30 paired cold-start
streams (1% malicious, 20,000 samples). Variants (reset Adam, the paper's
update rule, so the comparison with Table I is like-for-like):
  base         frozen model, static threshold (logit 0 == probability 0.5)
  aqt_prob     AQT on sigmoid outputs (the paper's rule)
  aqt_logit    AQT on pre-sigmoid logits
  oracle_logit true-label fine-tuning + logit AQT
  pseudo_logit confidence-gated pseudo-labels + logit AQT

usage: python3 controlled_feature_rerun.py <raw77|raw10> <shard> <nshards> [budget_s]
"""

import sys

from grid_runner import run_grid

SEEDS = [42, 123, 456, 789, 2024]
STREAMS_PER_SEED = 6

VARIANTS = {
    "base":         dict(space="logit", aqt=False, adapt="none", preseed="none"),
    "aqt_prob":     dict(space="prob",  aqt=True,  adapt="none"),
    "aqt_logit":    dict(space="logit", aqt=True,  adapt="none"),
    "oracle_logit": dict(space="logit", aqt=True,  adapt="oracle", opt="reset_adam"),
    "pseudo_logit": dict(space="logit", aqt=True,  adapt="pseudo", opt="reset_adam"),
}

if __name__ == "__main__":
    pool = sys.argv[1]
    shard, nshards = int(sys.argv[2]), int(sys.argv[3])
    budget = float(sys.argv[4]) if len(sys.argv) > 4 else None
    run_grid(f"controlled_{pool}", pool, VARIANTS, SEEDS, STREAMS_PER_SEED,
             shard=shard, nshards=nshards, time_budget=budget)
