#!/usr/bin/env bash
# Step 1 on the AWS instance: preflight, build 2 images, download pinned model cache. No GPU workload.
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p env hf-cache
nvidia-smi > env/nvidia-smi.txt; nvidia-smi -q > env/nvidia-smi-q.txt
docker info > env/docker-info.txt 2>&1
docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L > env/container-gpu-check.txt
T=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 300' || true)
for k in instance-type instance-id ami-id placement/region placement/availability-zone; do
  printf '%s=%s\n' "$k" "$(curl -s -H "X-aws-ec2-metadata-token: $T" http://169.254.169.254/latest/meta-data/$k || true)"; done > env/instance.txt
uname -a > env/uname.txt; df -h / > env/disk.txt; free -g > env/mem.txt
cat env/instance.txt; grep -E "Driver Version|Product Name|Total|Reserved" env/nvidia-smi-q.txt | head -6
docker build -q -f docker/Dockerfile.s0 -t outside-aws:s0 docker | tee env/image-s0.txt
docker build -q -f docker/Dockerfile.s4 -t outside-aws:s4 docker | tee env/image-s4.txt
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
PLAN=lab/qwen-bf16-plan.json
if [ "$VRAM" -lt 22000 ]; then
  python3 -c "import json;p=json.load(open('$PLAN'));p['cases']=[c for c in p['cases'] if 'qwen' not in c['case_id']];json.dump(p,open('lab/plan-small.json','w'),indent=2)"
  PLAN=lab/plan-small.json; echo "VRAM ${VRAM}MiB < 22000: GPT-2 only"
fi
echo "$PLAN" > env/plan-path.txt
docker run --rm -v "$PWD/lab":/lab:ro -v "$PWD/hf-cache":/hf -e HF_HOME=/hf -e NVIDIA_VISIBLE_DEVICES=void outside-aws:s0 \
  python /lab/prepare_cache.py --plan /lab/$(basename $PLAN) --revisions /lab/qwen-revisions.json
echo SETUP_OK
