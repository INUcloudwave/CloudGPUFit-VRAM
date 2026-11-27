#!/usr/bin/env bash
# GPU return (2026-11-11, RTX 3090) phase 1: import + order-preserving xMem replay + decision comparison.
# CPU request diff / hybrid replay is added where a complete CPU history exists (phase 2 re-runs the
# same command with --cpu-history once the per-step CPU traces finish). Runs alongside the CPU stage.
set -uo pipefail
EXP=/home/inuos/cgf-reserved-diag-20261110
cd "$EXP"
RET=inputs/gpu_return_20261111/return
LOG=results/gpu/import.log
mkdir -p results/gpu
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
for c in core__decoder_llm__small__fp16__01 core__decoder_llm__small__fp16__12 core__decoder_llm__medium__fp32__12; do
  for run in "$RET/$c"/*/; do
    run_id=$(basename "$run")
    [ -f "$run/memory_snapshot.pickle" ] || { say "skip $c $run_id (no snapshot)"; continue; }
    out=results/gpu/$c/$run_id
    [ -f "$out/import_report.json" ] && { say "skip $c $run_id (done)"; continue; }
    mkdir -p "$out"
    extra=()
    for h in runs/cpu_history/$c/from_config_per_step runs/cpu_history/$c/from_config; do
      if [ -f "$h/meta.json" ] && { [ -f "$h/DONE" ] || [ "$h" = "runs/cpu_history/$c/from_config" ]; }; then
        extra=(--cpu-history "/exp/$h"); break
      fi
    done
    say "import $c $run_id ${extra[*]}"
    MEM=16g THREADS=2 docs/run_in_image.sh -u -m cgf_diag.gpu_replay --run-dir "/exp/$run" --out "/exp/$out" \
        --capacity-gb 80 "${extra[@]}" > "$out/stdout.log" 2>&1
    say "  exit=$?"
  done
done
say "phase1 finished"
