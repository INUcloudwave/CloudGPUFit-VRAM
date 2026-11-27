#!/usr/bin/env bash
# Stage 2/3 on inuos after the inugpu return bundle arrives.
#   bash docs/run_gpu_import.sh /path/to/return_root            (run inside tmux)
# For every <case>/<run_id> directory: import (safe unpickler), replay into pinned xMem with the
# no-pressure capacity (80 GiB), compare decisions, diff against the CPU history trace, hybrid replay.
set -uo pipefail
EXP=/home/inuos/cgf-reserved-diag-20261110
RET=${1:?return_root}
cd "$EXP"
DEST=inputs/gpu_return_$(date +%Y%m%d)
mkdir -p "$DEST" results/gpu
# keep originals untouched: copy once, verify checksums if provided
rsync -a --ignore-existing "$RET"/ "$DEST"/
if [ -f "$DEST/SHA256SUMS" ]; then (cd "$DEST" && sha256sum -c SHA256SUMS > ../../results/gpu/return_sha256_check.txt 2>&1); fi
for run in "$DEST"/core__*/*/; do
  case_id=$(basename "$(dirname "$run")"); run_id=$(basename "$run")
  out=results/gpu/$case_id/$run_id
  mkdir -p "$out"
  hist=/exp/runs/cpu_history/$case_id/from_config
  echo "[$(date -Is)] import $case_id $run_id"
  docs/run_in_image.sh -u -m cgf_diag.gpu_replay --run-dir "/exp/$run" --out "/exp/$out" --capacity-gb 80 \
      --cpu-history "$hist" > "$out/stdout.log" 2>&1
  echo "  exit=$?"
done
docs/run_in_image.sh -u -m cgf_diag.summarize --out /exp/results/SUMMARY_after_gpu.md > results/gpu/summarize.log 2>&1
echo done
