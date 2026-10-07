#!/bin/bash
# Round 7 experiment queue: two processes at a time (2 cores).
cd /home/claude/kennethwatts/nids_atl
two() { "$@" & }
# A: clean-window study (both pools in parallel)
python3 -u round7_frozen.py cleanwin multi77 > r7_cw_multi77.log 2>&1 &
python3 -u round7_frozen.py cleanwin raw77 > r7_cw_raw77.log 2>&1 &
wait
# B: ranking dose-response
python3 -u round7_frozen.py dose multi77 > r7_dose_multi77.log 2>&1 &
python3 -u round7_frozen.py dose raw77 > r7_dose_raw77.log 2>&1 &
wait
# C-E: missing Table III cells
for pool in multi77 multi10 paper10; do
  python3 -u round7_grid.py cells $pool 0 2 > r7_cells_${pool}_0.log 2>&1 &
  python3 -u round7_grid.py cells $pool 1 2 > r7_cells_${pool}_1.log 2>&1 &
  wait
done
# F: held-out rate tuning (disjoint seeds)
for pool in multi77 paper10 multi10; do
  python3 -u round7_grid.py tune $pool 0 2 > r7_tune_${pool}_0.log 2>&1 &
  python3 -u round7_grid.py tune $pool 1 2 > r7_tune_${pool}_1.log 2>&1 &
  wait
done
# G: label-budget curve
python3 -u round7_grid.py labels multi77 0 2 > r7_labels_multi77_0.log 2>&1 &
python3 -u round7_grid.py labels multi77 1 2 > r7_labels_multi77_1.log 2>&1 &
wait
# H: fine-tuned replay
python3 -u round7_replay_ft.py multi77 42 > r7_rft_multi77_42.log 2>&1 &
python3 -u round7_replay_ft.py multi77 123 > r7_rft_multi77_123.log 2>&1 &
wait
python3 -u round7_replay_ft.py multi77 456 > r7_rft_multi77_456.log 2>&1 &
python3 -u round7_replay_ft.py raw77 42 > r7_rft_raw77_42.log 2>&1 &
wait
echo QUEUE_DONE > r7_queue_done.txt
