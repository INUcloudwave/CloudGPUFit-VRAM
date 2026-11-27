#!/bin/bash
# Compare A/C/B (+ previous g4dn L256 reference if present) and pack everything to bring back.
#   bash envcmp/collect.sh            -> results/envcmp/*, ~/cgf-envcmp-results-<host>-<ts>.tar.gz (+ .sha256)
set -uo pipefail
EXP=$(readlink -f "$(dirname "$0")/.."); cd "$EXP"; OUT=results/envcmp; mkdir -p $OUT
ARGS="A=runs/aws-g4dn.xlarge__envcmp_A C=runs/aws-g4dn.xlarge__envcmp_C B=runs/aws-g4dn.xlarge__envcmp_B"
[ -d runs/aws-g4dn.xlarge__l256 ] && ARGS="$ARGS REF=runs/aws-g4dn.xlarge__l256"   # previous g4dn L256 (39 selected), A reproducibility
python3 -I tools/compare_envs.py . config/conditions_envcmp39.json $OUT $ARGS | tee $OUT/compare_console.txt
TS=$(date +%Y%m%dT%H%M%S); T=$HOME/cgf-envcmp-results-$(hostname -s)-$TS.tar.gz
tar czf "$T" -C "$(dirname "$EXP")" $(cd "$(dirname "$EXP")" && ls -d "$(basename "$EXP")"/runs/aws-g4dn.xlarge__envcmp_* \
   "$(basename "$EXP")"/results/envcmp "$(basename "$EXP")"/envcmp/records "$(basename "$EXP")"/logs 2>/dev/null)
(cd "$(dirname "$T")" && sha256sum "$(basename "$T")" | tee "$(basename "$T").sha256")
echo "RESULT_TAR=$T"; tar tzf "$T" | wc -l
