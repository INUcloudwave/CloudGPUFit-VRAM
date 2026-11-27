#!/usr/bin/env bash
# Run a command inside the pinned CPU image used by the 2026-11-08 predictions.
# Mounts: xMem (read-only, lock-checked), production predictor worktree (read-only),
# this experiment directory (read-write). No network.
set -euo pipefail
EXP=${EXP:-/home/inuos/cgf-reserved-diag-20261110}
XMEM_WORK=${XMEM_WORK:-/home/inuos/xmem-work}
PREDICTOR=${PREDICTOR:-/home/inuos/CloudGPUFit-cpu-singlepass-20261108}
IMAGE=${IMAGE:-cloudgpufit-cpu-validation:20261108}
THREADS=${THREADS:-4}
MEM=${MEM:-36g}
NET=${NET:-none}
exec docker run --rm -i --network "$NET" --user "$(id -u):$(id -g)" --memory "$MEM" --memory-swap "$MEM" \
  --cpus "$THREADS" --tmpfs /tmp:exec,mode=1777 -e HOME=/tmp -e MPLCONFIGDIR=/tmp -e OMP_NUM_THREADS="$THREADS" -e MKL_NUM_THREADS="$THREADS" \
  -e OPENBLAS_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 -e HF_HOME=/exp/hf_cache \
  -e PYTHONPATH=/exp/src:/predictor:/work/xMem \
  --mount type=bind,src="$XMEM_WORK",dst=/work,readonly \
  --mount type=bind,src="$PREDICTOR",dst=/predictor,readonly \
  --mount type=bind,src="$EXP",dst=/exp \
  -w /exp --entrypoint python "$IMAGE" "$@"
