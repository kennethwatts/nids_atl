#!/bin/bash
cd /home/claude/kennethwatts/nids_atl
python3 -u round9_extras.py recover > r9b_recover.log 2>&1 &
python3 -u round9_extras.py prev 0 2 > r9b_prev0.log 2>&1 &
wait
python3 -u round9_extras.py prev 1 2 > r9b_prev1.log 2>&1
echo done > r9b_done.txt
