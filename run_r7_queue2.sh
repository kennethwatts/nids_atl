#!/bin/bash
cd /home/claude/kennethwatts/nids_atl
while [ ! -f r7_queue_done.txt ]; do sleep 30; done
python3 -u round7_grid.py gate paper10 0 2 > r7_gate_0.log 2>&1 &
python3 -u round7_grid.py gate paper10 1 2 > r7_gate_1.log 2>&1 &
wait
echo done > r7_queue2_done.txt
