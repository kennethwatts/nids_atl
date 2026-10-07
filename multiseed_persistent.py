"""
20-seed headline and persistent-optimizer grid with a finer rate grid
(Round 6 Reviewer 2 majors 2 and 3).

Reviewer 2: the persistent-optimizer results are single-seed (seed 42, the
seed most favourable to AQT) and the 10x Adam result is non-monotone, which
suggests instability or noise; five seeds is too few for a cluster bootstrap
and per-seed effects should be tested with a t-interval or 20+ seeds.

Setting: the paper's own 10-feature data (deduplicated), probability-space
AQT as in the paper, 20 pretraining seeds x 3 paired streams = 60 units of
20,000 samples at 1% malicious. The same units also give the 20-seed
deduplicated headline (base, aqt, oracle_reset, pseudo_reset). The
no-dedup headline is multiseed_headline_nodedup (same script, pool arg).

usage: python3 multiseed_persistent.py <paper10|paper10_nodedup> <shard> <nshards> [budget_s]
"""

import sys

from grid_runner import run_grid

SEEDS = [42, 123, 456, 789, 2024] + list(range(1000, 1015))
STREAMS = 1  # one stream per seed: seeds are the independent units (the first five seeds already have 3 streams)

HEAD = {
    "base":         dict(space="prob", aqt=False, adapt="none", preseed="none"),
    "aqt":          dict(space="prob", aqt=True, adapt="none"),
    "oracle_reset": dict(space="prob", aqt=True, adapt="oracle", opt="reset_adam"),
    "pseudo_reset": dict(space="prob", aqt=True, adapt="pseudo", opt="reset_adam"),
}
PERSIST = {}
for m in (1, 3, 10, 30, 100):
    PERSIST[f"oracle_adam_{m}x"] = dict(space="prob", aqt=True, adapt="oracle", opt="persist_adam", lr_mult=m)
for m in (1, 10):
    PERSIST[f"pseudo_adam_{m}x"] = dict(space="prob", aqt=True, adapt="pseudo", opt="persist_adam", lr_mult=m)
for m in (1, 30, 100):
    PERSIST[f"oracle_sgd_{m}x"] = dict(space="prob", aqt=True, adapt="oracle", opt="persist_sgd", lr_mult=m)

if __name__ == "__main__":
    pool = sys.argv[1]
    shard, nshards = int(sys.argv[2]), int(sys.argv[3])
    budget = float(sys.argv[4]) if len(sys.argv) > 4 else None
    v = dict(HEAD)
    if pool == "paper10":
        v.update(PERSIST)
    run_grid(f"multiseed_{pool}", pool, v, SEEDS, STREAMS, shard=shard, nshards=nshards, time_budget=budget)
