#!/usr/bin/env bash
set -e
cd /home/inugpu/cloudgpufit-outside
rm -rf pkgs/s2
for V in S3 S4; do v=$(echo $V | tr A-Z a-z)
  echo "== install $V $(date +%T)"; outside-lab/images/install_variant.sh $V > pkgs/install-$v.log 2>&1
  grep -E "^torch==|cublas-cu12|cudnn-cu12" pkgs/$v/environment.lock.txt
  echo "== run $V $(date +%T)"; python3 outside-lab/run_software.py $V --execute > software-$v-01.log 2>&1
  echo "== analyse $V"; python3 outside-lab/analyse_software.py $V > software-$v-01-analysis.log 2>&1
  python3 -I -c "import json;s=json.load(open('results/software-$v-01-analysis/summary.json'));print({k:s[k] for k in ('torch','primary_diff_range_mib','primary_all_within_100','unstable_plateau','max_repeat_range_mib')})"
  rm -rf pkgs/$v; df -h / | tail -1
done
echo ALL_DONE
