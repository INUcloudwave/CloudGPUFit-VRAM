#!/usr/bin/env bash
# Run a python command inside the pinned image on the AWS host (paths relative to this bundle). NET=1 allows network.
R=$(cd "$(dirname "$0")/../.." && pwd)            # bundle root: cgf-fp16-680-20261111/, xmem-work/, predictor/, diag_src/
E=$R/cgf-fp16-680-20261111
NETOPT="--network none"; [ "${NET:-0}" = 1 ] && NETOPT=""
exec docker run --rm -i $NETOPT --user "$(id -u):$(id -g)" --memory ${MEM:-16g} --cpus ${CPUS:-4} \
  --tmpfs /tmp:exec,mode=1777 -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e HF_HOME=/exp/hf_cache \
  $( [ "${NET:-0}" = 1 ] || echo "-e HF_HUB_OFFLINE=1" ) \
  -e PYTHONPATH=/exp/src:/diag/src:/predictor:/work/xMem \
  --mount type=bind,src=$R/xmem-work,dst=/work,readonly \
  --mount type=bind,src=$R/predictor,dst=/predictor,readonly \
  --mount type=bind,src=$R/diag_src,dst=/diag/src,readonly \
  --mount type=bind,src=$E,dst=/exp -w /exp --entrypoint python cloudgpufit-cpu-validation:20261108 "$@"
