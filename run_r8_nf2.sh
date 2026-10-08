#!/bin/bash
cd /home/claude/kennethwatts/nids_atl
export NF_SRC=/tmp/claude-0/-home-claude/aab038ab-ad45-5fe6-a0a2-c120a9e4fcbc/scratchpad/nf/NF-UNSW-NB15-v2.csv NF_TGT=/tmp/claude-0/-home-claude/aab038ab-ad45-5fe6-a0a2-c120a9e4fcbc/scratchpad/nf/tgt.csv
while [ ! -f r8_replay_done.txt ]; do sleep 60; done
for mode in r8 r8tune; do
  python3 -u round7_grid.py $mode nf 0 2 > r8_${mode}_nf_0.log 2>&1 &
  python3 -u round7_grid.py $mode nf 1 2 > r8_${mode}_nf_1.log 2>&1 &
  wait
done
echo done > r8_nf2_done.txt
