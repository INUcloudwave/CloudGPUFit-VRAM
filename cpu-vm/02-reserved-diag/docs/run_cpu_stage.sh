#!/usr/bin/env bash
# CPU-only stage of the reserved-error diagnosis. jobs=1 (strictly sequential). Run inside tmux:
#   tmux new-session -d -s cgf-diag "bash /home/inuos/cgf-reserved-diag-20261110/docs/run_cpu_stage.sh"
# Re-running skips steps whose DONE marker exists.
set -uo pipefail
EXP=/home/inuos/cgf-reserved-diag-20261110
cd "$EXP"
RUN="$EXP/docs/run_in_image.sh"
LOG="$EXP/runs/cpu_stage.log"
mkdir -p runs/baseline runs/cpu_history results/cpu
CASES=(core__decoder_llm__small__fp16__01 core__decoder_llm__medium__fp32__12 core__decoder_llm__small__fp16__12)
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
gpu_sha() { python3 -c "import json,glob;print(json.load(open(glob.glob('$EXP/inputs/aws_g6_2xlarge/results/*/$1__memory__r0.json')[0]))['input_sha256']['input_ids'])"; }

# 1. production baseline re-run (original predictor, unchanged) with the full chrome trace kept
for c in "${CASES[@]}"; do
  d=runs/baseline/$c; mkdir -p $d
  [ -f $d/DONE ] && { say "skip baseline $c"; continue; }
  say "baseline $c start"
  python3 -c "import json;s=json.load(open('inputs/cpu_production/$c.input.json'));r=dict(s['request']);r['trace_output']='/exp/$d/trace.json';print(json.dumps(r))" \
    | "$RUN" -u /predictor/predict_vram.py --json - > $d/prediction.json 2> $d/stderr.log
  rc=$?; say "baseline $c exit=$rc"
  if [ $rc -eq 0 ]; then gzip -1 -f $d/trace.json && touch $d/DONE; fi
done

# 2. GPU-equivalent history traces (AWS local_core replica: perf f1 w1 m5 -> memory f1 w1 m3)
for c in "${CASES[@]}"; do
  d=runs/cpu_history/$c/from_config; mkdir -p $d
  [ -f $d/DONE ] && { say "skip history $c"; continue; }
  say "history $c start"
  "$RUN" -u -m cgf_diag.cpu_history_trace --input-json inputs/cpu_production/$c.input.json \
     --sequence aws_replica_local_core --weights from_config --keep-raw-traces --out /exp/$d \
     --expected-input-sha "$(gpu_sha $c)" > $d/stdout.log 2> $d/stderr.log
  rc=$?; say "history $c exit=$rc"; [ $rc -eq 0 ] && touch $d/DONE
done

# 3. weights control (gpt2 only): pretrained weights single generate, compare request sequence
c=core__decoder_llm__small__fp16__01; d=runs/cpu_history/$c/pretrained_single; mkdir -p $d
if [ ! -f $d/DONE ] && [ -d hf_cache/hub/models--openai-community--gpt2 ]; then
  say "pretrained single $c start"
  "$RUN" -u -m cgf_diag.cpu_history_trace --input-json inputs/cpu_production/$c.input.json \
     --sequence single_generate --weights pretrained --out /exp/$d --expected-input-sha "$(gpu_sha $c)" > $d/stdout.log 2> $d/stderr.log
  rc=$?; say "pretrained single exit=$rc"; [ $rc -eq 0 ] && touch $d/DONE
fi

# 4. analyses
for c in "${CASES[@]}"; do
  say "analyze $c"
  "$RUN" -u -m cgf_diag.analyze_cpu --case $c --baseline-dir /exp/runs/baseline/$c \
     --history-dir /exp/runs/cpu_history/$c/from_config --out /exp/results/cpu/$c.analysis.json \
     > results/cpu/$c.analysis.stdout.log 2>&1
  say "analyze $c exit=$?"
done
"$RUN" -m cgf_diag.summarize --out /exp/results/SUMMARY_cpu.md > results/cpu/summarize.log 2>&1; say "summarize exit=$?"
say "CPU stage finished"
