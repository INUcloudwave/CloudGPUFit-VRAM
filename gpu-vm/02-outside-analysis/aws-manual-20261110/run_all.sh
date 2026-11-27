#!/usr/bin/env bash
# Run manually via nohup sudo -n bash run_all.sh g5.xlarge-01.
# Never starts/stops/terminates an EC2 instance and never upgrades host drivers.
set -euo pipefail
cd "$(dirname "$0")"
run_label=${1:-g5.xlarge-01}
if [[ ! "$run_label" =~ ^[[:alnum:]][[:alnum:]._-]*$ ]]; then printf 'Invalid label\n' >&2; exit 2; fi
if [[ -e .run-started ]]; then printf 'Previous attempt exists. Preserve it; use a fresh package directory for another attempt.\n' >&2; exit 2; fi
python3 verify_package.py
mkdir -p logs results env
( set -o noclobber; printf '%s\n' "$run_label" > .run-started )
finish() {
 final_code=$?
 trap - EXIT
 printf '%s\n' "$final_code" > run.exitcode
 date -u +'%Y-%m-%dT%H:%M:%SZ' > logs/finished-utc.txt
 stamp=$(date -u +'%Y%m%dT%H%M%SZ')
 tar -czf "results/${run_label}-execution-${stamp}.tar.gz" logs env run.exitcode package-manifest.json
 printf 'FINAL_EXIT_CODE=%s\n' "$final_code"
 printf 'Download results/*.tar.gz and run.exitcode, then STOP this instance in the EC2 console. No automatic stop.\n'
 exit "$final_code"
}
trap finish EXIT
bash setup.sh > logs/setup.log 2>&1
python3 run.py --label "$run_label" --dry-run > logs/schedule.json
python3 run.py --label "$run_label" > logs/experiment.log 2>&1
