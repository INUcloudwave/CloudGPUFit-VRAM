#!/usr/bin/env bash
# One (case, run_id) = one fresh container = one fresh process. Sequential use only.
# usage: run_one.sh <case_id> <run_id> [stacks(python|all)] [run_label_suffix]
set -u
R=/home/inugpu/cgf-reserved-diag-20261110
IMAGE_TAG=cloudgpufit-bench:cu124-v1
IMAGE_ID=sha256:457da255e6e2de7ee05ff7814674c90abcf6d1ed04e73df3cc4eebcdb31fb032
GPU=GPU-17eb7299-8b39-914d-7c5b-26908502904f
HF=/home/inugpu/cloudgpufit-bench/hf-cache
CASE=$1; RUN=$2; STACKS=${3:-python}; SUFFIX=${4:-}
LABEL=${RUN}${SUFFIX}
LOGS=$R/logs; mkdir -p "$LOGS"
L=$LOGS/${CASE}__${LABEL}
OUTDIR=$R/return/$CASE/$LABEL

if [ -e "$OUTDIR" ]; then echo "refuse: $OUTDIR exists" >&2; exit 90; fi
got=$(docker image inspect --format '{{.Id}}' $IMAGE_TAG) || exit 91
[ "$got" = "$IMAGE_ID" ] || { echo "image id mismatch $got" >&2; exit 92; }
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[ -z "$apps" ] || { echo "other GPU process present: $apps" >&2; exit 93; }
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,driver_version,temperature.gpu,pstate --format=csv > "$L.nvidia_smi_before.csv"

CMD=(docker run --rm --pull=never --network none --name "cgfdiag_$$" --gpus "device=$GPU" --user "$(id -u):$(id -g)"
  -e HOME=/tmp -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e CUDA_MODULE_LOADING=LAZY
  ${TORCH_SYMBOLIZE_MODE:+-e TORCH_SYMBOLIZE_MODE=$TORCH_SYMBOLIZE_MODE}
  -e CGF_IMAGE_ID=$IMAGE_ID -e CGF_IMAGE_TAG=$IMAGE_TAG -e CGF_HOST_LABEL=inugpu-rtx3090
  -v "$R/collector:/collector:ro" -v "$R/handoff:/handoff:ro" -v "$HF:/hf:ro" -v "$R/return:/results"
  -w /tmp --entrypoint python "$IMAGE_ID" -I /collector/gpu_collector.py
  --manifest /handoff/common_manifest.json --case "$CASE" --run-id "$RUN" --out /results --stacks "$STACKS")
[ -n "$SUFFIX" ] && CMD+=(--run-label-suffix "$SUFFIX")
printf '%q ' "${CMD[@]}" > "$L.cmd"; echo >> "$L.cmd"
# record the exact docker command inside the container env as well
CMD=("${CMD[@]:0:4}" -e "CGF_DOCKER_CMD=$(cat "$L.cmd")" "${CMD[@]:4}")

date -Is > "$L.start"
"${CMD[@]}" > "$L.stdout.log" 2> "$L.stderr.log"
rc=$?
date -Is > "$L.end"; echo $rc > "$L.exitcode"
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,driver_version,temperature.gpu,pstate --format=csv > "$L.nvidia_smi_after.csv"
if [ -d "$OUTDIR" ]; then
  for s in stdout.log stderr.log cmd start end exitcode nvidia_smi_before.csv nvidia_smi_after.csv; do cp -p "$L.$s" "$OUTDIR/$s"; done
fi
echo "$CASE $LABEL rc=$rc"
exit $rc
