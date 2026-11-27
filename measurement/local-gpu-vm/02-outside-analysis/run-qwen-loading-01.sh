#!/usr/bin/env bash
cd /home/inugpu/cloudgpufit-outside || exit 1
python3 outside-lab/run_qwen_loading.py --execute > qwen-loading-01.log 2>&1
qwen_loading_exit=$?
printf '%s\n' "$qwen_loading_exit" > qwen-loading-01.exitcode
exit "$qwen_loading_exit"
