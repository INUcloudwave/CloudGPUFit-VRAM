#!/usr/bin/env bash
# cuDNN workspace probe on one GPU (no model, no network). Same image as the VRAM measurement.
#   GPU_LABEL=rtx3090 IMAGE=cloudgpufit-bench:cu124-v1 bash run_probe.sh [extra args, e.g. --src model]
# Blackwell (g7, g7e): IMAGE=cloudgpufit-bench:cu128-v1
# SIGS=<file in this folder> selects another signature list (default conv_signatures.json).
# Output: ./probe_${GPU_LABEL}.jsonl (+ .console.log). Resumable: rerun the same command after an interruption.
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
IMAGE=${IMAGE:-cloudgpufit-bench:cu124-v1}
GPU_LABEL=${GPU_LABEL:?set GPU_LABEL, e.g. rtx3090 / aws-g6.xlarge}
SIGS=${SIGS:-conv_signatures.json}
IID=$(docker image inspect --format '{{.Id}}' "$IMAGE")
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader | tee "$D/nvidia-smi_${GPU_LABEL}.txt"
docker run --rm --gpus all --network none --user "$(id -u):$(id -g)" \
  -e CUDA_MODULE_LOADING=LAZY -e IMAGE_ID="$IID" -e GPU_LABEL="$GPU_LABEL" -e HOME=/tmp \
  -v "$D":/probe -w /probe --entrypoint python "$IMAGE" \
  /probe/cudnn_ws_probe.py --signatures "/probe/$SIGS" --out "/probe/probe_${GPU_LABEL}.jsonl" "$@" \
  2>&1 | tee -a "$D/probe_${GPU_LABEL}.console.log"
sha256sum "$D/probe_${GPU_LABEL}.jsonl"
