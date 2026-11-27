#!/usr/bin/env bash
# Pilot part A (models already downloaded), small->large, sequential, conv1d col-major (GPT-2 family only).
cd /home/inuos/cgf-fp16-680-20261111
python3 -u tools/run_suite.py --run-name pilot --select pilot --models opt-125m gpt2 gpt2-medium tinyllama-1.1b qwen2.5-1.5b \
  --threads 20 --mem 42g --conv1d-colmajor --timeout-hours 24 2>&1 | tee -a runs/pilot.console.log
