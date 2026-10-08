#!/bin/bash
# Round 8: second dataset family (NetFlow pair). Two processes at a time.
cd /home/claude/kennethwatts/nids_atl
export NF_SRC=/tmp/claude-0/-home-claude/aab038ab-ad45-5fe6-a0a2-c120a9e4fcbc/scratchpad/nf/NF-UNSW-NB15-v2.csv NF_TGT=/tmp/claude-0/-home-claude/aab038ab-ad45-5fe6-a0a2-c120a9e4fcbc/scratchpad/nf/tgt.csv
python3 -u round7_grid.py cells nf 0 2 > r8_cells_nf_0.log 2>&1 &
python3 -u round7_grid.py cells nf 1 2 > r8_cells_nf_1.log 2>&1 &
wait
python3 -u round7_grid.py tune nf 0 2 > r8_tune_nf_0.log 2>&1 &
python3 -u round7_grid.py tune nf 1 2 > r8_tune_nf_1.log 2>&1 &
wait
echo done > r8_nf_done.txt
