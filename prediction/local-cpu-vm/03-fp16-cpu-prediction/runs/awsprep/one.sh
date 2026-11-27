#!/usr/bin/env bash
# one.sh <case_id> <cases rel> <outdir rel> <threads> <cpuset> <mem> [worker args...]
E=/home/inuos/cgf-fp16-680-20261111
cid=$1; cases=$2; out=$3; th=$4; cs=$5; mem=$6; shift 6
mkdir -p $E/$out
docker run --rm --network none --user "$(id -u):$(id -g)" --memory $mem --memory-swap $mem --cpuset-cpus $cs \
  --tmpfs /tmp:exec,mode=1777 -e HOME=/tmp -e MPLCONFIGDIR=/tmp -e OMP_NUM_THREADS=$th -e MKL_NUM_THREADS=$th \
  -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=/exp/hf_cache \
  -e PYTHONPATH=/exp/src:/diag/src:/predictor:/work/xMem -e CGF_CPUSET=$cs -e CGF_MEM_LIMIT=$mem \
  --mount type=bind,src=/home/inuos/xmem-work,dst=/work,readonly \
  --mount type=bind,src=/home/inuos/CloudGPUFit-cpu-singlepass-20261108,dst=/predictor,readonly \
  --mount type=bind,src=/home/inuos/cgf-reserved-diag-20261110/src,dst=/diag/src,readonly \
  --mount type=bind,src=$E,dst=/exp -w /exp --entrypoint python cloudgpufit-cpu-validation:20261108 -u -m cgf680.cpu_worker2 \
  --case-id $cid --cases /exp/$cases --out /exp/$out --threads $th "$@" 2>&1 | grep CGF_RESULT | cut -c1-200
