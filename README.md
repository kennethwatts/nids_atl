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
| `extract_sample.py` | Order-preserving 1-in-k sampler for the large 2018 Tuesday file |
| `multiday_pool.py`, `multiday_sanity.py` | Multi-day pool (2017 Mon/Tue/Wed/Fri-DDoS/Fri-PortScan -> 2018 Wed 14, Tue 20, Wed 21 Feb), per-family frozen AUROC |
| `multiday_grid.py`, `summarize_multiday.py` | 20-seed headline and persistent-optimizer grid on the multi-day pool, 77 and 10 features (Table III) |
| `multiday_replay.py`, `summarize_multiday_replay.py` | Time-ordered replay of each multi-day 2018 day (Section V-B) |
| `round7_grid.py` (`cells`, `tune`, `labels`, `gate`), `flex_trial.py` (`label_frac`, `oracle_alert`), `summarize_round7.py`, `summarize_tuned.py`, `summarize_gate.py`, `summarize_table1.py` | Round 7: 20-seed cells for all four pools (budget, persistent pseudo-labels, Bonferroni flags), held-out rate tuning (seeds 3000-3005), label-budget curve, 20-seed gate check, seed-averaged Table I |
| `round7_frozen.py` (`cleanwin`, `dose`), `round7_tpr1.py`, `summarize_cleanwin.py`, `summarize_dose.py` | Round 7: clean-window threshold with 0/1/5% contamination, TPR at 1% FPR, ranking-noise dose-response |
| `round7_replay_ft.py`, `summarize_replayft.py` | Round 7: time-ordered replay of fine-tuned variants |
| `make_fig_r7.py`, `run_r7_queue*.sh` | Figure 1 of the paper and the experiment queue used for Round 7 |
| `nf_pool.py`, `run_r8_nf.sh`, `run_r8_nf2.sh` | Round 8: second dataset family (NF-UNSW-NB15-v2 source, NF-CSE-CIC-IDS2018-v2 target sample made with `extract_sample.py`); `NF_SRC`/`NF_TGT` env vars point at the files |
| `round7_grid.py` modes `r8`, `r8tune`, `flex_trial.py` (`oracle_benign`), `run_r8_queue.sh` | Round 8: reset Adam at 3x/10x, persistent-vs-reset at matched rates, pseudo-label rate grid and held-out tuning, benign-only labels |
| `round8_frozen.py`, `round7_frozen.py dose` | Round 8: R-precision next to TPR@1%FPR per pool and per multi-day day; dose-response with R-precision |
| `round8_robust.py`, `round8_surrogate.py`, `round8_replay_thr.py`, `summarize_replaythr.py`, `run_r8_frozen.sh`, `run_r8_replay.sh` | Round 8: robust clean-window quantiles and window drift, surrogate strength (disjoint data), fine-tuned replay under AQT/budget/clean-window rules |
| `round7_grid.py` modes `r9`, `r9tune`, `summarize_r9.py`, `run_r9.sh`, `run_r9_nf.sh` | Round 9: pseudo-label rate 300x (grid edge) on NF and multi-10, NF label-budget curve at 3x/10x |
| `R9=1 round8_robust.py`, `summarize_r9_robust.py` | Round 9: clean-window sweep to 20% contamination, contamination check and q' on held-out Tue 20 Feb (`round9_robust_summary.csv`) |

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
