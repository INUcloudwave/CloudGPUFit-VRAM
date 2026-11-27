#!/bin/bash
# RTX 3090 VM main run: conditions_main395_l256.json (1024->256), v2 (attn default), measure all -> validate all. Resumable (rerun = continue).
cd /home/inugpu/cgf-fp16-vram-20261111
LABEL=rtx3090__main395_l256
echo MAIN_START $(date -Iseconds)
for M in measure validate; do
  python3 -I src/run_conditions.py --conditions conditions_main395_l256.json --list all --mode $M --gpu-label $LABEL --attn default \
    || { echo RUNNER_ABORT $M $(date -Iseconds); exit 1; }
  echo ${M^^}_END $(date -Iseconds)
done
python3 -I tools/summarize_runs.py runs/$LABEL > logs/main395_${LABEL}_summary.md
echo MAIN_END $(date -Iseconds)
df -h / | tail -1
