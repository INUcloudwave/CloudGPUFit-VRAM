#!/usr/bin/env bash
cd /home/inugpu/cloudgpufit-outside || exit 1
python3 outside-lab/run_qwen_shape.py --execute > qwen-shape-01.log 2>&1
code=$?
printf '%s\n' "$code" > qwen-shape-01.exitcode
exit "$code"
