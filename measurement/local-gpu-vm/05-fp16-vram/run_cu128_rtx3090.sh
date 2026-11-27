#!/bin/bash
# 2026-11-17: version-effect baseline. Same GPU (RTX 3090), same runner/worker/conditions as rtx3090__main395_l256,
# only the image differs: cloudgpufit-bench:cu128-v1 (torch 2.7.1+cu128) instead of cu124-v1 (torch 2.6.0+cu124).
# pilot 26 (measure+validate+check) -> main 395 (measure+validate) only if pilot passed. Resumable.
cd /home/inugpu/cgf-fp16-vram-20261111
IMG=cloudgpufit-bench:cu128-v1; COND=conditions_main395_l256.json; LABEL=rtx3090__main395_l256__cu128
echo CU128_START $(date -Iseconds) image=$(docker image inspect --format '{{.Id}}' $IMG)
for M in measure validate; do
  python3 -I src/run_conditions.py --conditions $COND --list pilot --mode $M --gpu-label $LABEL --attn default --image $IMG \
    || { echo RUNNER_ABORT pilot $M; exit 1; }
done
python3 -I tools/check_pilot.py . $LABEL config/pilot.json --conditions $COND; RC=$?
echo PILOT_END rc=$RC $(date -Iseconds)
[ $RC = 0 ] || { echo STOP pilot not passed; exit 1; }
date -Iseconds > runs/$LABEL/PILOT_PASSED
for M in measure validate; do
  python3 -I src/run_conditions.py --conditions $COND --list all --mode $M --gpu-label $LABEL --attn default --image $IMG \
    || { echo RUNNER_ABORT main $M; exit 1; }
  echo ${M^^}_END $(date -Iseconds)
done
python3 -I tools/summarize_runs.py runs/$LABEL > logs/main395_${LABEL}_summary.md
echo CU128_END $(date -Iseconds); df -h / | tail -1
