#!/bin/bash
# Run one environment over the 39 conditions (measure: load -> warmup 3 -> measure 5, one fresh container per condition),
# then the attention-kernel diagnostic (separate processes, not a measurement). Resumable: rerun = continue.
#   bash envcmp/run_env.sh <A|C|B> <HF_CACHE_DIR>
set -uo pipefail
EXP=$(readlink -f "$(dirname "$0")/.."); cd "$EXP"; ENV=${1:?A|C|B}; CACHE=$(readlink -f "${2:?hf cache dir}")
case $ENV in A|B) IMG=cloudgpufit-bench:cu124-v1 ;; C) IMG=cloudgpufit-bench:pt251-cu124-v1 ;; *) echo "ABORT env $ENV"; exit 1 ;; esac
COND=conditions_envcmp39.json; LABEL=aws-g4dn.xlarge__envcmp_$ENV; mkdir -p logs; LOG=logs/envcmp_$ENV.txt
{
echo "ENV_START $ENV $(date -Iseconds) image=$IMG label=$LABEL"
V=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1); echo "driver=$V kernel=$(uname -r)"
A_FACTS=$(ls -d envcmp/records/A_before_*/DRIVER_FACTS.txt 2>/dev/null | head -1)
if [ "$ENV" != A ]; then
  [ -n "$A_FACTS" ] || { echo "ABORT: run A first"; exit 1; }
  AV=$(grep ^driver= "$A_FACTS" | cut -d= -f2); AK=$(grep ^kernel= "$A_FACTS" | cut -d= -f2); AP=$(grep ^persistence= "$A_FACTS" | cut -d= -f2)
  [ "$(uname -r)" = "$AK" ] || { echo "ABORT: kernel $(uname -r) != A kernel $AK"; exit 1; }
  [ "$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader)" = "$AP" ] || { echo "ABORT: persistence mode differs from A ($AP). Fix: sudo nvidia-smi -pm $([ "$AP" = Enabled ] && echo 1 || echo 0)"; exit 1; }
  if [ "$ENV" = C ]; then [ "$V" = "$AV" ] || { echo "ABORT: C needs A's driver $AV, found $V"; exit 1; }
  else [[ $V == 580.* && $V != "$AV" ]] || { echo "ABORT: B needs an R580 driver, found $V"; exit 1; }; fi
fi
bash envcmp/host_record.sh ${ENV}_before | tail -8
python3 -I tools/run_conditions_env.py --conditions $COND --list all --mode measure --gpu-label $LABEL --attn default \
  --hf-cache "$CACHE" --image $IMG --env-name $ENV || { echo "RUNNER_ABORT"; exit 1; }
echo "MEASURE_END $(date -Iseconds)"
mkdir -p runs/$LABEL/_diag
if [ ! -f runs/$LABEL/_diag/attention_kernels.json ]; then
  IDS=$(python3 -I -c "import json;c=json.load(open('config_diag/conditions.json'))['diag_ids'];print(' '.join(c))")
  test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" && docker run --rm --pull=never --network none \
    --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e CUDA_MODULE_LOADING=LAZY \
    -v "$CACHE:/hf:ro" -v "$EXP/tools:/tools:ro" -v "$EXP/config_diag:/config:ro" -v "$EXP/runs/$LABEL/_diag:/out" -w /tmp \
    --entrypoint python $IMG -I /tools/diag_attention_kernels.py /config /out/attention_kernels.json $IDS \
    > runs/$LABEL/_diag/attention_kernels.log 2>&1 || echo "KERNEL_DIAG_FAILED (diagnostic only; measurement unaffected)"
fi
python3 -I -c "
import json,glob,collections;c=collections.Counter(json.load(open(p)).get('status') for p in glob.glob('runs/$LABEL/*/measure.json'));print('STATUS',dict(c))"
echo "ENV_END $ENV $(date -Iseconds)"
} 2>&1 | tee -a "$LOG"
