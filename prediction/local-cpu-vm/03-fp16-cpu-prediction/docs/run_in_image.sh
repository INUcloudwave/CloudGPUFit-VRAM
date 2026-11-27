#!/usr/bin/env bash
# Run a python command inside the pinned CPU image (analysis helpers). No network.
E=/home/inuos/cgf-fp16-680-20261111
exec docker run --rm -i --network none --user "$(id -u):$(id -g)" --memory ${MEM:-16g} --cpus ${CPUS:-4} \
  --tmpfs /tmp:exec,mode=1777 -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 -e HF_HOME=/exp/hf_cache \
  -e PYTHONPATH=/exp/src:/diag/src:/predictor:/work/xMem \
  --mount type=bind,src=/home/inuos/xmem-work,dst=/work,readonly \
  --mount type=bind,src=/home/inuos/CloudGPUFit-cpu-singlepass-20261108,dst=/predictor,readonly \
  --mount type=bind,src=/home/inuos/cgf-reserved-diag-20261110/src,dst=/diag/src,readonly \
  --mount type=bind,src=$E,dst=/exp -w /exp --entrypoint python cloudgpufit-cpu-validation:20261108 "$@"
