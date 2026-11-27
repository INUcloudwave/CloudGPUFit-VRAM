#!/usr/bin/env bash
# Re-create the AWS bundle image file from the local image (same image ID -> same content as the 2026-11-17 bundle).
#   bash docs/image/export_image.sh /path/to/outdir
set -euo pipefail
IMG=cloudgpufit-cpu-validation:20261108
WANT=sha256:875290b5854ad6dbc8ea554fecb292dd5ffdfea2a03e8c4e8179a3c827968fa5
OUT=${1:?outdir}; mkdir -p "$OUT"
GOT=$(docker image inspect $IMG --format '{{.Id}}')
[ "$GOT" = "$WANT" ] || { echo "image id mismatch: $GOT (want $WANT)"; exit 1; }
docker save $IMG | zstd -T8 -10 -q -o "$OUT/cloudgpufit-cpu-validation_20261108.tar.zst"
( cd "$OUT" && sha256sum cloudgpufit-cpu-validation_20261108.tar.zst )
echo "load check: zstd -dc <file> | docker load  ->  docker image inspect $IMG --format '{{.Id}}' must print $WANT"
