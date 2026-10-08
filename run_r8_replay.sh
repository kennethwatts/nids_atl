#!/bin/bash
cd /home/claude/kennethwatts/nids_atl
while [ ! -f r8_queue_done.txt ]; do sleep 60; done
python3 -u round8_replay_thr.py multi77 42 > r8_rt_m42.log 2>&1 &
python3 -u round8_replay_thr.py multi77 123 > r8_rt_m123.log 2>&1 &
wait
python3 -u round8_replay_thr.py multi77 456 > r8_rt_m456.log 2>&1 &
python3 -u round8_replay_thr.py raw77 42 > r8_rt_r42.log 2>&1 &
wait
echo done > r8_replay_done.txt
