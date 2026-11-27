#!/usr/bin/env bash
cd /home/inugpu/cloudgpufit-outside || exit 1
python3 outside-lab/run_local_closeout.py --execute > local-closeout-20261110.log 2>&1
closeout_exit=$?
printf '%s\n' "$closeout_exit" > local-closeout-20261110.exitcode
exit "$closeout_exit"
