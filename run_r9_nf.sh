#!/bin/bash
cd /home/claude/kennethwatts/nids_atl
export NF_SRC=/tmp/claude-0/-home-claude/aab038ab-ad45-5fe6-a0a2-c120a9e4fcbc/scratchpad/nf/NF-UNSW-NB15-v2.csv NF_TGT=/tmp/claude-0/-home-claude/aab038ab-ad45-5fe6-a0a2-c120a9e4fcbc/scratchpad/nf/tgt.csv
python3 -u round7_grid.py r9 nf 0 2 > r9_r9_nf_0.log 2>&1
while pgrep -f "round7_grid.py r9 multi10" >/dev/null; do sleep 10; done
python3 -u round7_grid.py r9 nf 1 2 > r9_r9_nf_1.log 2>&1
echo done > r9_done.txt
