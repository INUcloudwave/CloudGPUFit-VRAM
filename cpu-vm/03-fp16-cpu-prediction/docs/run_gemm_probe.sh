#!/usr/bin/env bash
# reserved prediction v3: GEMM probe for one model (B=1, small generation), in the pinned CPU image, no network.
#   docs/run_gemm_probe.sh <case_id> [input_len]   -> runs/gemm_probe/<case_id>[__inN]/gemm_probe.json
set -euo pipefail
E=/home/inuos/cgf-fp16-680-20261111
CASE=$1; IL=${2:-}
OUT=runs/gemm_probe/${CASE}${IL:+__in$IL}__pb${PB:-1}__pn${PN:-2}
mkdir -p "$E/$OUT"
exec docker run --rm --network none --user "$(id -u):$(id -g)" --memory ${MEM:-24g} --cpus ${CPUS:-4} \
  --tmpfs /tmp:exec,mode=1777 -e HOME=/tmp -e MPLCONFIGDIR=/tmp -e OMP_NUM_THREADS=4 -e PYTHONDONTWRITEBYTECODE=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=/exp/hf_cache \
  -e PYTHONPATH=/exp/src:/diag/src:/predictor:/work/xMem \
  --mount type=bind,src=/home/inuos/xmem-work,dst=/work,readonly \
  --mount type=bind,src=/home/inuos/CloudGPUFit-cpu-singlepass-20261108,dst=/predictor,readonly \
  --mount type=bind,src=/home/inuos/cgf-reserved-diag-20261110/src,dst=/diag/src,readonly \
  --mount type=bind,src=$E,dst=/exp -w /exp --entrypoint python cloudgpufit-cpu-validation:20261108 \
  -u -m cgf680.gemm_probe --case-id "$CASE" ${IL:+--input-len $IL} --batch ${PB:-1} --new-tokens ${PN:-2} --out "/exp/$OUT"
