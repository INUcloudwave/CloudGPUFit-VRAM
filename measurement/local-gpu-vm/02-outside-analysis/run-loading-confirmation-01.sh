#!/usr/bin/env bash
cd /home/inugpu/cloudgpufit-outside || exit 1
python3 outside-lab/run_loading_confirmation.py --execute > loading-confirmation-01.log 2>&1
confirmation_exit=$?
printf '%s\n' "$confirmation_exit" > loading-confirmation-01.exitcode
exit "$confirmation_exit"
