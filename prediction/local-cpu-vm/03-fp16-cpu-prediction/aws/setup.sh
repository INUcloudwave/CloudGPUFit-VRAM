#!/usr/bin/env bash
# One-time setup on the AWS instance. Run from the bundle root:  bash cgf-fp16-680-20261111/aws/setup.sh <image.tar.zst>
set -euo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd); E=$R/cgf-fp16-680-20261111
IMG=${1:?image tar.zst}
echo "== host"; nproc; free -g | head -2; df -h "$R" | tail -1
grep -o -m1 'avx512_fp16' /proc/cpuinfo >/dev/null && echo "CPU flag avx512_fp16: yes" || echo "CPU flag avx512_fp16: NO (fp16_real will be slow)"
grep -o -m1 'amx_fp16' /proc/cpuinfo >/dev/null && echo "CPU flag amx_fp16: yes" || echo "CPU flag amx_fp16: no"
command -v docker >/dev/null || { echo "install docker first (see README_AWS_KO.md)"; exit 1; }
command -v tmux >/dev/null || echo "WARN: tmux missing (sudo dnf/apt install tmux)"
echo "== load image"; zstd -dc "$IMG" | docker load
docker image inspect cloudgpufit-cpu-validation:20261108 --format '{{.Id}}'
echo "== download models (network, ~28 GB)"; mkdir -p "$E/hf_cache"
NET=1 MEM=16g CPUS=8 bash "$E/aws/in_image.sh" -u aws/download_models.py | tail -3
echo "== xMem lock check"
MEM=4g CPUS=1 bash "$E/aws/in_image.sh" -c "from cgf_diag.replay import verify_xmem; print(verify_xmem('/work/xMem','/exp/provenance/xmem-allocator-lock.json')['ok'])"
echo "== smoke test (3 small conditions, fp16_real)"
cd "$E" && python3 tools/run_aws.py --run-name aws_smoke --ids opt-125m__b1__in128__new128 resnet-18__b1__r128 t5-small__b1__in128__new128 \
   --worker-args "--dtype-mode fp16_real" | tail -2
echo "SETUP_DONE"
