#!/usr/bin/env bash
# Pilot 26 (subset of the 395) into the SAME run directory as the main run -> never re-run in main.
cd /home/inuos/cgf-fp16-680-20261111
python3 -u tools/run395.py --run-name main395 --select pilot --phases A B 2>&1 | tee -a runs/main395.pilot.console.log
