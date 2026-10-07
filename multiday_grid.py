"""
20-seed headline and persistent-optimizer grid on the multi-day 2017 -> 2018 pool
(Round 6 Reviewer 1, major 1). Logit-space AQT (probabilities saturate at 77 features).
usage: python3 multiday_grid.py <multi77|multi10> <shard> <nshards> [budget_s]
"""
import sys
from grid_runner import run_grid

SEEDS = [42, 123, 456, 789, 2024] + list(range(1000, 1015))
V = {
    "base":         dict(space="logit", aqt=False, adapt="none", preseed="none"),
    "aqt":          dict(space="logit", aqt=True, adapt="none"),
    "oracle_reset": dict(space="logit", aqt=True, adapt="oracle", opt="reset_adam"),
    "pseudo_reset": dict(space="logit", aqt=True, adapt="pseudo", opt="reset_adam"),
}
for m in (1, 10, 30):
    V[f"oracle_adam_{m}x"] = dict(space="logit", aqt=True, adapt="oracle", opt="persist_adam", lr_mult=m)
V["pseudo_adam_10x"] = dict(space="logit", aqt=True, adapt="pseudo", opt="persist_adam", lr_mult=10)

if __name__ == "__main__":
    pool, shard, nshards = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    budget = float(sys.argv[4]) if len(sys.argv) > 4 else None
    run_grid(f"multiday_{pool}", pool, V, SEEDS, 1, shard=shard, nshards=nshards, time_budget=budget)
