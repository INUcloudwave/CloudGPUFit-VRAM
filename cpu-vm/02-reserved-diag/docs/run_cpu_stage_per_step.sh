#!/usr/bin/env bash
# CPU history traces with ONE PROCESS PER STEP (fix for the 2nd-profiler-session RSS blow-up).
# Order: fp16_01 (equivalence check vs the completed single-process trace) -> medium fp32 -> fp16_12
#        -> pretrained-weights check (gpt2) -> analyses -> summary. jobs=1. Run inside tmux.
# Re-running resumes: finished steps (steps/<KK>/meta.json) and DONE markers are skipped.
set -uo pipefail
EXP=/home/inuos/cgf-reserved-diag-20261110
cd "$EXP"
RUN="$EXP/docs/run_in_image.sh"
LOG="$EXP/runs/cpu_stage.log"
say() { echo "[$(date -Is)] per-step: $*" | tee -a "$LOG"; }
gpu_sha() { python3 -c "import json,glob;print(json.load(open(glob.glob('$EXP/inputs/aws_g6_2xlarge/results/*/$1__memory__r0.json')[0]))['input_sha256']['input_ids'])"; }
NSTEPS=13   # aws_replica_local_core: 1+1+5+1+1+1+3

trace_case() {  # case, outdir, [compare_dir]
  local c=$1 d=$2 cmp=${3:-}
  mkdir -p "$d"
  if [ -f "$d/DONE" ]; then say "skip $c (DONE)"; return 0; fi
  local sha; sha=$(gpu_sha "$c")
  for k in $(seq 0 $((NSTEPS-1))); do
    kk=$(printf %02d "$k")
    if [ -f "$d/steps/$kk/meta.json" ]; then continue; fi
    mkdir -p "$d/steps/$kk"
    "$RUN" -u -m cgf_diag.cpu_history_trace --input-json inputs/cpu_production/$c.input.json \
       --sequence aws_replica_local_core --weights from_config --keep-raw-traces --step-index "$k" \
       --mem-log /exp/$d/steps/$kk/mem.jsonl --out /exp/$d --expected-input-sha "$sha" \
       > "$d/steps/$kk/stdout.log" 2> "$d/steps/$kk/stderr.log"
    rc=$?; say "$c step $kk exit=$rc"
    if [ $rc -ne 0 ]; then say "$c stopped at step $kk"; return 1; fi
  done
  if [ -n "$cmp" ]; then
    "$RUN" -u -m cgf_diag.merge_steps --out /exp/$d --compare-with /exp/$cmp > "$d/merge.log" 2>&1
  else
    "$RUN" -u -m cgf_diag.merge_steps --out /exp/$d > "$d/merge.log" 2>&1
  fi
  rc=$?; say "$c merge exit=$rc $(tail -1 "$d/merge.log")"
  [ $rc -eq 0 ] && touch "$d/DONE"
}

analyze() {  # case, history dir
  "$RUN" -u -m cgf_diag.analyze_cpu --case "$1" --baseline-dir /exp/runs/baseline/$1 \
     --history-dir /exp/$2 --out /exp/results/cpu/$1.analysis.json > results/cpu/$1.analysis.stdout.log 2>&1
  say "analyze $1 exit=$?"
}

C1=core__decoder_llm__small__fp16__01; C2=core__decoder_llm__medium__fp32__12; C3=core__decoder_llm__small__fp16__12
trace_case $C1 runs/cpu_history/$C1/from_config_per_step runs/cpu_history/$C1/from_config
trace_case $C2 runs/cpu_history/$C2/from_config_per_step && analyze $C2 runs/cpu_history/$C2/from_config_per_step
trace_case $C3 runs/cpu_history/$C3/from_config_per_step && analyze $C3 runs/cpu_history/$C3/from_config_per_step

d=runs/cpu_history/$C1/pretrained_single
if [ ! -f $d/DONE ]; then
  mkdir -p $d
  "$RUN" -u -m cgf_diag.cpu_history_trace --input-json inputs/cpu_production/$C1.input.json \
     --sequence single_generate --weights pretrained --out /exp/$d --expected-input-sha "$(gpu_sha $C1)" > $d/stdout.log 2> $d/stderr.log
  rc=$?; say "pretrained single exit=$rc"; [ $rc -eq 0 ] && touch $d/DONE
fi
"$RUN" -m cgf_diag.summarize --out /exp/results/SUMMARY_cpu.md > results/cpu/summarize.log 2>&1
say "summarize exit=$?"
say "per-step stage finished"
