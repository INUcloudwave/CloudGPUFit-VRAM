#!/usr/bin/env bash
# Retry of history traces that did not finish in run_cpu_stage.sh (2026-11-10: medium fp32 exit=137,
# suspected profiler-result leak while automatic GC was disabled; tracer fixed to collect outside
# profiled windows). Waits for the first stage to end so that only one model job runs at a time.
set -uo pipefail
EXP=/home/inuos/cgf-reserved-diag-20261110
cd "$EXP"
RUN="$EXP/docs/run_in_image.sh"
LOG="$EXP/runs/cpu_stage.log"
say() { echo "[$(date -Is)] retry: $*" | tee -a "$LOG"; }
gpu_sha() { python3 -c "import json,glob;print(json.load(open(glob.glob('$EXP/inputs/aws_g6_2xlarge/results/*/$1__memory__r0.json')[0]))['input_sha256']['input_ids'])"; }
say "waiting for first stage to finish"
until grep -q "CPU stage finished" "$LOG"; do sleep 30; done
mkdir -p runs/failed
for c in core__decoder_llm__medium__fp32__12 core__decoder_llm__small__fp16__12; do
  d=runs/cpu_history/$c/from_config
  if [ -f $d/DONE ]; then say "skip $c (DONE)"; continue; fi
  if [ -d $d ]; then mv $d runs/failed/${c}_from_config_$(date +%H%M%S); say "moved partial $c to runs/failed"; fi
  mkdir -p $d
  say "history $c start"
  "$RUN" -u -m cgf_diag.cpu_history_trace --input-json inputs/cpu_production/$c.input.json \
     --sequence aws_replica_local_core --weights from_config --keep-raw-traces --out /exp/$d \
     --expected-input-sha "$(gpu_sha $c)" > $d/stdout.log 2> $d/stderr.log
  rc=$?; say "history $c exit=$rc"; [ $rc -eq 0 ] && touch $d/DONE
  say "analyze $c"
  "$RUN" -u -m cgf_diag.analyze_cpu --case $c --baseline-dir /exp/runs/baseline/$c \
     --history-dir /exp/$d --out /exp/results/cpu/$c.analysis.json > results/cpu/$c.analysis.stdout.log 2>&1
  say "analyze $c exit=$?"
done
"$RUN" -m cgf_diag.summarize --out /exp/results/SUMMARY_cpu.md > results/cpu/summarize.log 2>&1
say "retry finished"
