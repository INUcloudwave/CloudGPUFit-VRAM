#!/bin/bash
# Build (download at pinned revisions) and/or verify the 13-model HF cache.
#   bash aws/aws_02_cache.sh download <HF_CACHE_DIR>   # needs internet (huggingface.co); then verifies with --hash
#   bash aws/aws_02_cache.sh verify   <HF_CACHE_DIR>   # e.g. after streaming the cache from INUGPU
set -euo pipefail
MODE=$1; CACHE=$(readlink -f "$2"); EXP=$(readlink -f "$(dirname "$0")/..")
mkdir -p "$CACHE"
IMG=${CGF_IMAGE_TAG:-cloudgpufit-bench:cu124-v1}   # 2026-11-17: image variant from RUN_ON_AWS.sh
if [ "$MODE" = download ]; then
  docker run --rm --pull=never --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/hf -v "$CACHE:/hf" -v "$EXP:/exp:ro" \
    -w /tmp --entrypoint python $IMG -I /exp/tools/prepare_cache.py /exp/config
fi
docker run --rm --pull=never --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/hf -v "$CACHE:/hf:ro" \
  -v "$EXP:/exp:ro" -w /tmp --entrypoint python $IMG -I /exp/tools/prepare_cache.py /exp/config --verify-only --hash \
  | tee "$EXP/logs/cache_verify_$(hostname)_$(date +%Y%m%dT%H%M%S).txt"
