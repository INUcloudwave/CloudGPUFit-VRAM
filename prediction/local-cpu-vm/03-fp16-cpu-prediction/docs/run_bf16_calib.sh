#!/usr/bin/env bash
# After FP32 calibration: BF16 proxy timing calibration on the 26 pilot shapes (sequential, 20 threads).
cd /home/inuos/cgf-fp16-680-20261111
while tmux has-session -t cgf680-fp32cal 2>/dev/null; do sleep 30; done
python3 -u tools/run_suite.py --run-name bf16-calib --select all --cases-file manifest/bf16_calib_cases.jsonl \
  --threads 20 --mem 42g --timeout-hours 3 2>&1 | tee -a runs/bf16-calib.console.log
python3 tools/collect.py --run-name bf16-calib
