#!/bin/bash
cd /home/claude/kennethwatts/nids_atl
nice -n 10 python3 -u round8_robust.py multi77 > r8_robust_m.log 2>&1
nice -n 10 python3 -u round8_robust.py raw77 > r8_robust_r.log 2>&1
