"""
Pseudo-label arm with a persistent optimizer, higher-rate sweep, and
negatives-only / positives-only ablations (Round 5 Reviewer 2 major 1-3).

Reviewer 2: (1) the persistent-optimizer experiment covered the oracle only,
so every "pseudo-label stays slightly below AQT" claim still rests on the
per-step reset optimizer; rerun pseudo-labels with persistent Adam and SGD at
1x and 10x. (2) Persistent Adam at 10x reached 0.561, above AQT's 0.518 and
the largest gap in the paper; sweep rates above 10x for the oracle and
pseudo-labels. (3) Only 4.6% of positive pseudo-labels are correct: add a
negatives-only and a positives-only ablation, and use bias-drift
instrumentation to explain why pseudo-labels beat the oracle under the reset
optimizer.

Setting: the paper's own 10-feature data (deduplicated), pretraining seed 42,
probability-space AQT exactly as in the paper (the 10-feature model has no
score saturation: base_model_sanity_check.py), 30 paired streams of 20,000
samples at 1% malicious. All variants share the same streams.

Variants (lr multiplier is relative to BASE_LR = 1.38e-3):
  base, aqt                     reference
  oracle_reset, pseudo_reset    the paper's rule
  {oracle,pseudo}_adam_{1,10,30,100}x   persistent Adam
  {oracle,pseudo}_sgd_{1,10,100}x       persistent SGD
  pseudoneg_{reset,adam1x}, pseudopos_{reset,adam1x}   one-sided pseudo-labels

usage: python3 pseudo_persistent_ablation.py <shard> <nshards> [budget_s]
"""

import sys

from grid_runner import run_grid, shard_args

SEEDS = [42]
STREAMS = 30

V = {
    "base": dict(space="prob", aqt=False, adapt="none", preseed="none"),
    "aqt": dict(space="prob", aqt=True, adapt="none"),
    "oracle_reset": dict(space="prob", aqt=True, adapt="oracle", opt="reset_adam", trace_every=1000),
    "pseudo_reset": dict(space="prob", aqt=True, adapt="pseudo", opt="reset_adam", trace_every=1000),
    "pseudoneg_reset": dict(space="prob", aqt=True, adapt="pseudo_neg", opt="reset_adam", trace_every=1000),
    "pseudopos_reset": dict(space="prob", aqt=True, adapt="pseudo_pos", opt="reset_adam", trace_every=1000),
    "pseudoneg_adam1x": dict(space="prob", aqt=True, adapt="pseudo_neg", opt="persist_adam", lr_mult=1),
    "pseudopos_adam1x": dict(space="prob", aqt=True, adapt="pseudo_pos", opt="persist_adam", lr_mult=1),
}
for m in (1, 10, 30, 100):
    V[f"oracle_adam_{m}x"] = dict(space="prob", aqt=True, adapt="oracle", opt="persist_adam", lr_mult=m)
    V[f"pseudo_adam_{m}x"] = dict(space="prob", aqt=True, adapt="pseudo", opt="persist_adam", lr_mult=m)
for m in (1, 10, 100):
    V[f"oracle_sgd_{m}x"] = dict(space="prob", aqt=True, adapt="oracle", opt="persist_sgd", lr_mult=m)
    V[f"pseudo_sgd_{m}x"] = dict(space="prob", aqt=True, adapt="pseudo", opt="persist_sgd", lr_mult=m)

if __name__ == "__main__":
    shard, nshards, budget = shard_args()
    run_grid("pseudo_persistent", "paper10", V, SEEDS, STREAMS, shard=shard, nshards=nshards, time_budget=budget)
