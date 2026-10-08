#!/bin/bash
# Round 8 experiments, after the NF queue: r8 (fairness grid) and r8tune on the four existing pools.
cd /home/claude/kennethwatts/nids_atl
while [ ! -f r8_nf_done.txt ]; do sleep 30; done
for pool in multi77 multi10 raw77 paper10; do
  python3 -u round7_grid.py r8 $pool 0 2 > r8_r8_${pool}_0.log 2>&1 &
  python3 -u round7_grid.py r8 $pool 1 2 > r8_r8_${pool}_1.log 2>&1 &
  wait
done
for pool in multi77 multi10 raw77 paper10; do
  python3 -u round7_grid.py r8tune $pool 0 2 > r8_r8tune_${pool}_0.log 2>&1 &
  python3 -u round7_grid.py r8tune $pool 1 2 > r8_r8tune_${pool}_1.log 2>&1 &
  wait
done
echo done > r8_queue_done.txt
