"""
Round 7 grids on the same seeds and stream construction as multiseed_persistent.py
and multiday_grid.py (seed*1000+j streams, 20 pretraining seeds, 1 stream each),
so every new variant pairs with the earlier runs. Variants may carry
special="budget": the causal top-1% rule on the frozen model's scores.

usage: python3 round7_grid.py <mode> <pool> <shard> <nshards> [budget_s]
modes:
  cells   missing Table III cells (top-1% budget, persistent pseudo rows, extra rates;
          raw77 = Wednesday pair gets the full variant set at 20 seeds)
  tune    rate tuning on DISJOINT seeds 3000-3005 (oracle persistent Adam rates)
  labels  label-budget curve: oracle persistent Adam with labels on a random
          fraction of flows, and on alerted flows only
"""
import json, os, sys, time
import numpy as np
import torch

from verified_pipeline import pretrain_source_model, make_cold_start_stream
from flex_trial import run_flex_trial, metrics_at, score_buffer, model_logit
from grid_runner import load_pool
from threshold_baselines import budget_preds

torch.set_num_threads(1)
SEEDS20 = [42, 123, 456, 789, 2024] + list(range(1000, 1015))
TUNE_SEEDS = list(range(3000, 3006))
SPACE = {"paper10": "prob", "paper10_nodedup": "prob"}  # raw/multi pools use logits


def variants_for(mode, pool):
    sp = SPACE.get(pool, "logit")
    V = {}
    if mode == "cells":
        V["base"] = dict(space=sp, aqt=False, adapt="none", preseed="none")
        V["aqt"] = dict(space=sp, aqt=True, adapt="none")
        V["budget_1pct"] = dict(special="budget")
        if pool == "raw77":
            V["oracle_reset"] = dict(space=sp, aqt=True, adapt="oracle", opt="reset_adam")
            V["pseudo_reset"] = dict(space=sp, aqt=True, adapt="pseudo", opt="reset_adam")
            for m in (1, 3, 10, 30, 100):
                V[f"oracle_adam_{m}x"] = dict(space=sp, aqt=True, adapt="oracle", opt="persist_adam", lr_mult=m)
            for m in (1, 10):
                V[f"pseudo_adam_{m}x"] = dict(space=sp, aqt=True, adapt="pseudo", opt="persist_adam", lr_mult=m)
        elif pool in ("multi77", "multi10"):
            V["pseudo_adam_1x"] = dict(space=sp, aqt=True, adapt="pseudo", opt="persist_adam", lr_mult=1)
            for m in (3, 100):
                V[f"oracle_adam_{m}x"] = dict(space=sp, aqt=True, adapt="oracle", opt="persist_adam", lr_mult=m)
    elif mode == "tune":
        V["aqt"] = dict(space=sp, aqt=True, adapt="none")
        for m in (1, 3, 10, 30, 100):
            V[f"oracle_adam_{m}x"] = dict(space=sp, aqt=True, adapt="oracle", opt="persist_adam", lr_mult=m)
    elif mode == "gate":  # Round 7 R2-1: 20-seed versions of the single-seed gate diagnostics (paper pair)
        V["aqt"] = dict(space=sp, aqt=True, adapt="none")
        V["oracle_reset"] = dict(space=sp, aqt=True, adapt="oracle", opt="reset_adam")
        V["oracle_gated_reset"] = dict(space=sp, aqt=True, adapt="oracle_gated", opt="reset_adam")
        V["oracle_rand518_reset"] = dict(space=sp, aqt=True, adapt="oracle", opt="reset_adam", label_frac=0.518)
    elif mode == "labels":
        V["aqt"] = dict(space=sp, aqt=True, adapt="none")
        for m in (1, 10):
            for f in (0.001, 0.01, 0.05, 0.1, 0.25, 1.0):
                V[f"lab{f}_adam_{m}x"] = dict(space=sp, aqt=True, adapt="oracle", opt="persist_adam", lr_mult=m, label_frac=f)
            V[f"alert_adam_{m}x"] = dict(space=sp, aqt=True, adapt="oracle_alert", opt="persist_adam", lr_mult=m)
    return V


def main():
    mode, pool, shard, nshards = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    budget = float(sys.argv[5]) if len(sys.argv) > 5 else None
    seeds = TUNE_SEEDS if mode == "tune" else SEEDS20
    variants = variants_for(mode, pool)
    ck = f"round7_{mode}_{pool}_shard{shard}.json"
    done = json.load(open(ck)) if os.path.exists(ck) else {}
    X17, y17, X18, y18 = load_pool(pool)
    dim = X17.shape[1]
    t0 = time.time()
    # paper10 headline units: the first five seeds also have streams 1 and 2 in the earlier 20-seed run
    units = [(sd, j) for sd in seeds for j in (range(3) if (pool == "paper10" and mode == "cells" and sd in SEEDS20[:5]) else range(1))]
    for u, (seed, jj) in enumerate(units):
        key = f"{seed}_{jj}"
        if u % nshards != shard or key in done:
            continue
        torch.manual_seed(seed)
        m = pretrain_source_model(X17, y17, dim)
        state = m.state_dict()
        bufs = {sp: score_buffer(m, X17, sp) for sp in ("prob", "logit")}
        rng = np.random.RandomState(seed * 1000 + jj)
        Xs, ys = make_cold_start_stream(X18, y18, 20000, 0.01, rng)
        rec = {}
        for vname, kw in variants.items():
            kw = dict(kw)
            if kw.pop("special", None) == "budget":
                m.eval()
                with torch.no_grad():
                    s = model_logit(m, torch.tensor(Xs, dtype=torch.float32)).numpy().flatten()
                res = dict(preds=budget_preds(s, 0.01, 0.0), y=ys.astype(np.float32))
                r = metrics_at(res)
            else:
                pre = kw.pop("preseed", "auto")
                space = kw.get("space", "logit")
                res = run_flex_trial(state, dim, Xs, ys, 20000,
                                     preseed=(bufs[space] if pre == "auto" else None), **kw)
                r = metrics_at(res)
                r["n_updates"] = res["n_updates"]
                r["lab_stats"] = res["lab_stats"]
            rec[vname] = r
        done[key] = rec
        json.dump(done, open(ck, "w"))
        print(f"  [{mode} {pool} shard {shard}] seed={seed} done ({len(done)} units, {time.time()-t0:.0f}s)", flush=True)
        if budget is not None and time.time() - t0 >= budget:
            print("Time budget reached; re-run to continue."); return


if __name__ == "__main__":
    main()
