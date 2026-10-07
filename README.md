# nids_atl

Code, data pointers and saved results for *Is Fine-Tuning Needed for NIDS
Cold-Start? Adaptive Quantile Thresholding versus Label-Driven Adaptation*
(Watts and Vokkarane, ICC 2027).

Every quantitative result in the paper is produced by a script in this
repository. Experiments that fine-tune a model are checkpointed after every
Monte Carlo run (or every paired stream) and resume when re-invoked.

## Data

- `data/robust_2017_final.csv`, `data/robust_2018_final.csv`: the paper's
  10-feature source and target samples (provenance of the original raw days
  is not recorded).
- Raw CICFlowMeter day files supplied for the Round 5 experiments (not in
  the repository, too large): `Wednesday-workingHours.pcap_ISCX.csv`
  (CIC-IDS2017) and `Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv`
  (CSE-CIC-IDS2018). `raw_pool.py` cleans, harmonizes the two column-naming
  conventions (77 shared features) and caches the pool.

## Core pipeline

| Script | Produces |
|---|---|
| `verified_pipeline.py` | Table I (cold-start ablation), model, AQT, pseudo-label rules |
| `precision_recall_analysis.py` | Attack-class precision/recall/PR-AUC (Table III), infiltration metrics |
| `diagnose_oracle.py` | Bias-compression diagnostic (Section IV-C.2) |
| `oracle_robustness.py` | Oracle learning-rate and class-weight checks |
| `persistent_optimizer_oracle.py` | Oracle with a persistent optimizer (Adam, SGD; 4 rates) |
| `aqt_quantile_sensitivity.py` | AQT quantile sweep |
| `infiltration_case_study.py`, `logit_quantile_infiltration.py` | Infiltration case study, probability vs logit AQT |

## Round 4 additions

| Script | Question answered |
|---|---|
| `static_and_budget_baselines.py` | Static source threshold, causal top-1% budget, no pre-seed |
| `prevalence_q_burst_sweep.py` | Prevalence sweep, q up to 0.999, 20% burst |
| `statistical_rigor.py` | Effect sizes, Holm and Bonferroni, 5-seed AQT effect, pool bootstrap |
| `pseudolabel_diagnostics.py` | Pseudo-label precision, 50%-skip oracle |
| `snapshot_prauc_dedup_sensitivity.py` | Snapshot PR-AUC, no-dedup sensitivity |
| `adaptive_attacker_init_window.py` | Initialization-window attacker (no pre-seed) |
| `all_features_rerun.py` | First 77-feature rerun (superseded: see below) |

## Round 5 additions

| Script | Question answered |
|---|---|
| `base_model_sanity_check.py` | 77-input architecture, in-domain and per-family AUROC, probability vs logit AUROC, column alignment, per-feature shift |
| `flex_trial.py`, `grid_runner.py`, `summarize_grid.py` | Shared trial runner (validated bit-identical to `verified_pipeline.run_one_trial`), sharded paired-stream grid, cluster-bootstrap summaries |
| `controlled_feature_rerun.py` | Same pool, only the feature set changes (10 vs 77), logit-space AQT |
| `pseudo_persistent_ablation.py` | Pseudo-labels with persistent Adam/SGD, rates up to 100x, one-sided pseudo-labels |
| `oracle_gated_ablation.py` | True labels restricted to the pseudo-label gate |
| `seed_headline.py` | Seed-averaged Table I on the deduplicated and no-dedup pools |
| `threshold_baselines.py` | Source-calibrated static threshold, mismatched-prevalence budgets, hybrid cap, window length, attack-class metrics |
| `adversarial_preseeded.py` | Decoy inflation and slow-drift poisoning against the pre-seeded default |
| `time_ordered_replay.py` | Time-ordered replay of the raw 2018 day file |
| `summarize_decoy.py`, `summarize_baselines.py`, `summarize_replay.py` | Summaries for the decoy, baseline and replay tables |
| `round6_frozen.py` (`decoy`, `mitig`, `bbse`), `summarize_round6.py` | Random/surrogate/victim-aware decoys (Table IV), clean-window mitigations, BBSE, prevalence-matched q |
| `multiseed_persistent.py`, `summarize_multiseed.py` | 20-seed grid with persistent optimizers, t-intervals across seeds (Table II) |
| `time_ordered_finetune.py`, `summarize_tof.py` | Fine-tuned variants under time-ordered replay |

`all_features_rerun.py` (commit `2c985b5`) ran AQT on sigmoid outputs, 93.5%
of which were exactly 0.0 for its pretraining seed; its AUROC and its
oracle/pseudo/AQT tie are artifacts. `base_model_sanity_check.py` documents
the problem and `controlled_feature_rerun.py` replaces it.

## Running

```
python3 verified_pipeline.py                       # Table I
python3 controlled_feature_rerun.py raw77 0 2 &    # two shards in parallel
python3 controlled_feature_rerun.py raw77 1 2 &
python3 summarize_controlled.py
```

Sharded scripts take `<shard> <nshards> [time_budget_seconds]` and resume from
their `*_shard<k>.json` checkpoints.
