#!/usr/bin/env bash
# User-executed on an already running EC2 GPU instance. No driver/package upgrade.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p env hf-cache results
for cmd in nvidia-smi docker python3 curl; do command -v "$cmd" >/dev/null; done
python3 verify_package.py
nvidia-smi > env/nvidia-smi.txt
nvidia-smi -q > env/nvidia-smi-q.txt
cat /etc/os-release > env/os-release.txt
uname -a > env/uname.txt
df -h / > env/disk-before.txt
free -g > env/mem.txt
python3 - <<'PY'
import shutil,subprocess
assert len(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines())==1,'Use a single-GPU instance'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Other GPU processes are running'
assert shutil.disk_usage('.').free>=60*2**30,'Need at least 60GiB free for images/cache/build temporary data'
PY
docker info > env/docker-info.txt 2>&1
imds_token=$(curl -fsS --connect-timeout 2 --max-time 5 -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 300')
for item in instance-type instance-id ami-id placement/region placement/availability-zone; do
  item_value=$(curl -fsS --connect-timeout 2 --max-time 5 -H "X-aws-ec2-metadata-token: $imds_token" "http://169.254.169.254/latest/meta-data/$item")
  printf '%s=%s\n' "$item" "$item_value"
done > env/instance.txt
unset imds_token
cat env/instance.txt
docker build -q -f docker/Dockerfile.s0 -t outside-aws:s0 docker | tee env/image-s0.txt
docker build -q -f docker/Dockerfile.s4 -t outside-aws:s4 docker | tee env/image-s4.txt
docker run --rm --pull=never --network none --gpus all outside-aws:s0 nvidia-smi -L > env/container-gpu-check.txt
for image in s0 s4; do
 docker run --rm --pull=never --network none -e NVIDIA_VISIBLE_DEVICES=void "outside-aws:$image" python -m pip freeze > "env/pip-freeze-$image.txt"
done
vram_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
plan_file=lab/qwen-bf16-plan.json
if [ "$vram_mib" -lt 22000 ]; then
 python3 - <<'PY'
import json
from pathlib import Path
p=json.loads(Path('lab/qwen-bf16-plan.json').read_text());p['cases']=[c for c in p['cases'] if 'qwen' not in c['case_id']];Path('lab/plan-small.json').write_text(json.dumps(p,indent=2))
PY
 plan_file=lab/plan-small.json
 printf 'VRAM %sMiB <22000: GPT-2 only\n' "$vram_mib"
fi
printf '%s\n' "$plan_file" > env/plan-path.txt
docker run --rm --pull=never -v "$PWD/lab":/lab:ro -v "$PWD/hf-cache":/hf -e HF_HOME=/hf -e NVIDIA_VISIBLE_DEVICES=void outside-aws:s0 python /lab/prepare_cache.py --plan "/lab/$(basename "$plan_file")" --revisions /lab/qwen-revisions.json
df -h / > env/disk-after-setup.txt
printf 'SETUP_OK\n'
