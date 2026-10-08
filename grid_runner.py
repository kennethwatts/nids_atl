"""
Sharded, checkpointed grid runner for the Round 5 experiments.

A "unit" is one (pretraining seed, stream index) pair. For every unit, every
variant is run on the SAME resampled cold-start stream (paired design), with
the model pretrained once per seed. Units are dealt round-robin to shards so
two processes can run in parallel on the 2-core machine, each with its own
checkpoint file (<name>_shard<k>.json). Re-invoking resumes from the
checkpoint; a time budget (seconds) stops cleanly between units.

Pools:
  paper10         the paper's 10-feature robust_201{7,8}_final.csv, deduplicated
  paper10_nodedup the same files without deduplication
  raw77 / raw10   the 2017/2018 raw day files (raw_pool.py), 77 features or the
                  paper's ten pipeline features
Variants are dicts of run_flex_trial keyword arguments, plus
  preseed: 'auto' (source buffer in the variant's score space) or 'none'.
"""

import json
import os
import sys
import time

import numpy as np
import torch

from verified_pipeline import SEED, pretrain_source_model, make_cold_start_stream
from flex_trial import run_flex_trial, metrics_at, score_buffer


def load_pool(name):
    if name == "paper10":
        from verified_pipeline import load_data
        X17, y17, X18, y18, _ = load_data()
        return X17, y17, X18, y18
    if name == "paper10_nodedup":
        from snapshot_prauc_dedup_sensitivity import load_data_no_dedup
        X17, y17, X18, y18, _, _ = load_data_no_dedup()
        return X17, y17, X18, y18
    if name in ("raw77", "raw10"):
        from raw_pool import get_pool
        p = get_pool(name[3:])
        return p["X17"], p["y17"], p["X18"], p["y18"]
    if name in ("multi77", "multi10"):
        from multiday_pool import get_pool
        p = get_pool(name[5:])
        return p["X17"], p["y17"], p["X18"], p["y18"]
    if name == "nf":
        from nf_pool import get_pool
        p = get_pool()
        return p["X17"], p["y17"], p["X18"], p["y18"]
    raise ValueError(name)


def run_grid(name, pool, variants, seeds, streams_per_seed, n=20000, rate=0.01,
             shard=0, nshards=1, time_budget=None, extra_per_trial=None):
    ck_path = f"{name}_shard{shard}.json"
    done = {}
    if os.path.exists(ck_path):
        done = json.load(open(ck_path))
    X17, y17, X18, y18 = load_pool(pool)
    dim = X17.shape[1]
    t0 = time.time()
    models = {}
    units = [(s, j) for s in seeds for j in range(streams_per_seed)]
    for u, (seed, j) in enumerate(units):
        key = f"{seed}_{j}"
        if u % nshards != shard or key in done:
            continue
        if seed not in models:
            torch.manual_seed(seed)
            m = pretrain_source_model(X17, y17, dim)
            models[seed] = (m.state_dict(), {sp: score_buffer(m, X17, sp) for sp in ("prob", "logit")})
        state, bufs = models[seed]
        rng = np.random.RandomState(seed * 1000 + j)
        Xs, ys = make_cold_start_stream(X18, y18, n, rate, rng)
        rec = {}
        for vname, kw in variants.items():
            kw = dict(kw)
            pre = kw.pop("preseed", "auto")
            space = kw.get("space", "logit")
            res = run_flex_trial(state, dim, Xs, ys, n,
                                 preseed=(bufs[space] if pre == "auto" else None), **kw)
            r = metrics_at(res)
            r["n_updates"] = res["n_updates"]
            r["lab_stats"] = res["lab_stats"]
            if res["trace"]:
                r["trace"] = res["trace"]
            if extra_per_trial:
                r.update(extra_per_trial(res, Xs, ys))
            rec[vname] = r
        done[key] = rec
        json.dump(done, open(ck_path, "w"))
        el = time.time() - t0
        print(f"  [{name} shard {shard}/{nshards}] unit seed={seed} stream={j} done "
              f"({len(done)} units total, {el:.0f}s this invocation)", flush=True)
        if time_budget is not None and el >= time_budget:
            print("Time budget reached; re-run to continue.")
            return False
    return True


def load_all(name, nshards):
    out = {}
    for k in range(nshards):
        p = f"{name}_shard{k}.json"
        if os.path.exists(p):
            out.update(json.load(open(p)))
    return out


def shard_args():
    """argv: shard nshards time_budget"""
    shard = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nshards = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    budget = float(sys.argv[3]) if len(sys.argv) > 3 else None
    return shard, nshards, budget
