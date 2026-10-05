"""
Seed-averaged headline table, on both the deduplicated and the raw
(no-dedup) pools (Round 5 Reviewer 1 major concern 3 and Reviewer 2 major 5).

Reviewer 1: "The headline numbers come from the most favorable seed, and the
favorable dedup choice. Put seed means and ranges in Table I and the
abstract, and show both pools." Reviewer 2: "The p-values in Table II still
come from one seed. With five seeds, test the effect across seeds or use a
mixed model, and report effect sizes."

Setting: the paper's own 10-feature data, probability-space AQT and the
paper's reset-Adam update rule (the paper's exact configurations), 5
pretraining seeds x 6 streams = 30 paired cold-start streams per pool, 20,000
samples at 1% malicious. Reported with attack-class metrics too, since
macro-F1 sits near the always-benign floor (0.4975) here.

usage: python3 seed_headline.py <paper10|paper10_nodedup> <shard> <nshards> [budget_s]
"""

import sys

from grid_runner import run_grid

SEEDS = [42, 123, 456, 789, 2024]
STREAMS_PER_SEED = 6

VARIANTS = {
    "base":          dict(space="prob", aqt=False, adapt="none", preseed="none"),
    "aqt":           dict(space="prob", aqt=True,  adapt="none"),
    "unfreeze":      dict(space="prob", aqt=False, adapt="oracle", opt="reset_adam", preseed="none"),
    "oracle_reset":  dict(space="prob", aqt=True,  adapt="oracle", opt="reset_adam"),
    "pseudo_reset":  dict(space="prob", aqt=True,  adapt="pseudo", opt="reset_adam"),
}

if __name__ == "__main__":
    pool = sys.argv[1]
    shard, nshards = int(sys.argv[2]), int(sys.argv[3])
    budget = float(sys.argv[4]) if len(sys.argv) > 4 else None
    run_grid(f"seedhead_{pool}", pool, VARIANTS, SEEDS, STREAMS_PER_SEED,
             shard=shard, nshards=nshards, time_budget=budget)
