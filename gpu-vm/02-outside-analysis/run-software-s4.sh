#!/usr/bin/env bash
set -e
cd /home/inugpu/cloudgpufit-outside
echo "== install S4 $(date +%T)"; outside-lab/images/install_variant.sh S4 > pkgs/install-s4.log 2>&1
grep -E "^torch==|cublas-cu12|cudnn-cu12" pkgs/s4/environment.lock.txt
echo "== run S4 $(date +%T)"; python3 outside-lab/run_software.py S4 --execute > software-s4-01.log 2>&1
echo "== analyse S4"; python3 outside-lab/analyse_software.py S4 > software-s4-01-analysis.log 2>&1
python3 -I -c "import json;s=json.load(open('results/software-s4-01-analysis/summary.json'));print({k:s[k] for k in ('torch','primary_diff_range_mib','primary_all_within_100','unstable_plateau','max_repeat_range_mib')})"
rm -rf pkgs/s4; df -h / | tail -1; echo ALL_DONE
