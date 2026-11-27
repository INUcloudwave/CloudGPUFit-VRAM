#!/usr/bin/env bash
# FP32 timing calibration on the 26 pilot shapes (sequential, 20 threads, same tracer). No Conv1D change.
cd /home/inuos/cgf-fp16-680-20261111
python3 -u tools/run_suite.py --run-name fp32-calib --select all --cases-file manifest/fp32_calib_cases.jsonl \
  --threads 20 --mem 42g --timeout-hours 3 2>&1 | tee -a runs/fp32-calib.console.log
python3 tools/collect.py --run-name fp32-calib
