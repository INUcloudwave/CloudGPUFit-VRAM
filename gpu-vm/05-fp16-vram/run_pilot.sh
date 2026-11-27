#!/usr/bin/env bash
# RTX 3090 pilot: 26 measure -> 26 validate (separate processes) -> 5 trace
cd /home/inugpu/cgf-fp16-vram-20261111
echo "PILOT_START $(date -Is)"
python3 -I src/run_conditions.py --list pilot --mode measure --gpu-label rtx3090 || echo "MEASURE_RUNNER_FAILED"
echo "MEASURE_END $(date -Is)"
python3 -I src/run_conditions.py --list pilot --mode validate --gpu-label rtx3090 || echo "VALIDATE_RUNNER_FAILED"
echo "VALIDATE_END $(date -Is)"
python3 -I src/run_conditions.py --list config/trace_pilot.json --mode trace --gpu-label rtx3090 || echo "TRACE_RUNNER_FAILED"
echo "PILOT_DONE $(date -Is)"; df -h / | tail -1
