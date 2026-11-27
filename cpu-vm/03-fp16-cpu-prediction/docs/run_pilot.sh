#!/usr/bin/env bash
# Pilot: 26 cases, cost order (small first), sequential, 20 threads, 42g RAM limit, USER_SCOPE profiler,
# Conv1D col-major (GPT-2 family only; request-identical, verified). Resume: re-run this script.
cd /home/inuos/cgf-fp16-680-20261111
python3 -u tools/run_suite.py --run-name pilot --select pilot --threads 20 --mem 42g --conv1d-colmajor \
  --timeout-hours 24 2>&1 | tee -a runs/pilot.console.log
python3 tools/collect.py --run-name pilot 2>&1 | tee -a runs/pilot.console.log
