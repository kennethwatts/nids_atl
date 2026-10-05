"""
Oracle restricted to the pseudo-label gate (Round 5 Reviewer 2 major 3).

Reviewer 2: fewer updates is ruled out as the reason pseudo-labels beat the
oracle under the reset optimizer, "so the cause is still open". Two things
differ between the oracle and the pseudo-label arm: WHICH samples are updated
(the oracle updates every sample; the pseudo arm only the ~52% the confidence
gate selects, 98.8% of them confident-negatives) and WHICH labels are used
(true vs. self-assigned). This ablation separates them: 'oracle_gated' uses
TRUE labels but updates only on the samples the pseudo-label gate selects.

Same setting as pseudo_persistent_ablation.py (paper 10-feature data,
pretraining seed 42, 30 paired streams, reset Adam, probability-space AQT),
so its results are directly comparable to oracle_reset, pseudo_reset,
pseudoneg_reset and aqt there. Bias/score trace every 1,000 samples.

usage: python3 oracle_gated_ablation.py <shard> <nshards> [budget_s]
"""
from grid_runner import run_grid, shard_args

V = {
    "aqt": dict(space="prob", aqt=True, adapt="none"),
    "oracle_reset": dict(space="prob", aqt=True, adapt="oracle", opt="reset_adam", trace_every=1000),
    "oracle_gated_reset": dict(space="prob", aqt=True, adapt="oracle_gated", opt="reset_adam", trace_every=1000),
    "pseudo_reset": dict(space="prob", aqt=True, adapt="pseudo", opt="reset_adam", trace_every=1000),
}

if __name__ == "__main__":
    shard, nshards, budget = shard_args()
    run_grid("oracle_gated", "paper10", V, [42], 30, shard=shard, nshards=nshards, time_budget=budget)
