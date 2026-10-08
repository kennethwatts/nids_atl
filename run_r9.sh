#!/bin/bash
cd /home/claude/kennethwatts/nids_atl
export NF_SRC=/tmp/claude-0/-home-claude/aab038ab-ad45-5fe6-a0a2-c120a9e4fcbc/scratchpad/nf/NF-UNSW-NB15-v2.csv NF_TGT=/tmp/claude-0/-home-claude/aab038ab-ad45-5fe6-a0a2-c120a9e4fcbc/scratchpad/nf/tgt.csv
(R9=1 python3 -u round8_robust.py multi77 > r9_robust_m.log 2>&1; R9=1 python3 -u round8_robust.py raw77 > r9_robust_r.log 2>&1) &
for job in "r9tune nf" "r9tune multi10" "r9 multi10" "r9 nf"; do
  set -- $job
  python3 -u round7_grid.py $1 $2 0 1 > r9_$1_$2.log 2>&1
done
wait
echo done > r9_done.txt
