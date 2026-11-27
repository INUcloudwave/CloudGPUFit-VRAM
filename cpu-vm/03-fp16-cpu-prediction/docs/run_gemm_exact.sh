#!/usr/bin/env bash
# reserved prediction v3: exact GEMM probe (stop at first GEMM) for a list of cases of ONE table, pinned CPU image.
#   docs/run_gemm_exact.sh <cases.jsonl rel> <npz root rel> <out.jsonl rel> <case ids: a,b,c | @file rel>
set -euo pipefail
E=/home/inuos/cgf-fp16-680-20261111
IDS=$4; [[ $IDS == @* ]] && IDS=@/exp/${IDS#@}
exec docker run --rm --network none --user "$(id -u):$(id -g)" --memory ${MEM:-40g} --memory-swap ${MEM:-40g} --cpus ${CPUS:-4} \
  --tmpfs /tmp:exec,mode=1777 -e HOME=/tmp -e MPLCONFIGDIR=/tmp -e OMP_NUM_THREADS=${CPUS:-4} -e PYTHONDONTWRITEBYTECODE=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=/exp/hf_cache \
  -e PYTHONPATH=/exp/src:/diag/src:/predictor:/work/xMem \
  --mount type=bind,src=/home/inuos/xmem-work,dst=/work,readonly \
  --mount type=bind,src=/home/inuos/CloudGPUFit-cpu-singlepass-20261108,dst=/predictor,readonly \
  --mount type=bind,src=/home/inuos/cgf-reserved-diag-20261110/src,dst=/diag/src,readonly \
  --mount type=bind,src=$E,dst=/exp -w /exp --entrypoint python cloudgpufit-cpu-validation:20261108 \
  -u -m cgf680.gemm_probe --exact --cases /exp/$1 --npz-root /exp/$2 --out /exp/$3 --threads ${CPUS:-4} --abort-after ${ABORT_AFTER:-8} --case-ids "$IDS"
