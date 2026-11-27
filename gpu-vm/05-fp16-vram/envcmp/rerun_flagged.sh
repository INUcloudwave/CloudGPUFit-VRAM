#!/bin/bash
# Re-run only the flagged conditions in the CURRENT environment, each in a new process, into a separate label
# (original results are never touched).   bash envcmp/rerun_flagged.sh <A|C|B> <flagged json> <round 1|2|...> <HF_CACHE_DIR>
#   A/C: run while A's driver is still installed (before switching to B). B: run while the R580 driver is installed.
set -uo pipefail
EXP=$(readlink -f "$(dirname "$0")/.."); cd "$EXP"; ENV=$1; LIST=$(readlink -f "$2"); R=$3; CACHE=$(readlink -f "$4")
case $ENV in A|B) IMG=cloudgpufit-bench:cu124-v1 ;; C) IMG=cloudgpufit-bench:pt251-cu124-v1 ;; *) exit 1 ;; esac
python3 -I tools/run_conditions_env.py --conditions conditions_envcmp39.json --list "$LIST" --mode measure \
  --gpu-label aws-g4dn.xlarge__envcmp_${ENV}__rerun$R --attn default --hf-cache "$CACHE" --image $IMG --env-name ${ENV}_rerun$R \
  2>&1 | tee -a logs/envcmp_${ENV}_rerun$R.txt
