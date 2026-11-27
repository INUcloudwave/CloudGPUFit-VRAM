#!/usr/bin/env bash
# Main run after the pilot: remaining 369 (pilot results in the same run dir are skipped) + retry phase C.
cd /home/inuos/cgf-fp16-680-20261111
while tmux has-session -t cgf395-pilot 2>/dev/null; do sleep 60; done
python3 tools/collect395.py --run-name main395 > results/main395/pilot_collect.log 2>&1
cp results/main395/pilot_vs_rtx3090.csv results/main395/pilot_vs_rtx3090.after_pilot.csv 2>/dev/null
python3 -u tools/run395.py --run-name main395 --select all --phases A B C 2>&1 | tee -a runs/main395.main.console.log
python3 tools/collect395.py --run-name main395 2>&1 | tee -a runs/main395.main.console.log
