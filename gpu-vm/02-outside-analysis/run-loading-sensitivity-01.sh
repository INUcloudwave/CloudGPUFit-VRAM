#!/usr/bin/env bash
cd /home/inugpu/cloudgpufit-outside || exit 1
python3 outside-lab/prepare_loading_sensitivity.py --execute --output results/loading-sensitivity-01 > loading-sensitivity-01.log 2>&1
loading_exit=$?
printf '%s\n' "$loading_exit" > loading-sensitivity-01.exitcode
exit "$loading_exit"
