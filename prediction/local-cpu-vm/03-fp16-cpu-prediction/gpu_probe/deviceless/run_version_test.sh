#!/usr/bin/env bash
# cuDNN version test + deviceless check on one GPU (2026-11-23). Measurement worker, images and old results untouched.
#   bash run_version_test.sh            (IMAGE default cloudgpufit-bench:cu124-v1 = torch 2.6.0, cuDNN 9.1.0)
# 1) probe all 4,206 conv signatures with the image's own cuDNN 9.1.0
# 2) for each other cuDNN version: download the PyPI wheel (network only for this pip download, sha256 checked),
#    install into ./site_<v>, run the SAME probe with the SAME torch 2.6 but that libcudnn (LD_PRELOAD), delete the copy
# 3) cuDNN 9.28: save device properties of GPU 0, run the deviceless query (plans from properties, no handle)
# Output: probe_rtx3090_cudnn<v>.jsonl, props_rtx3090_cudnn9.28.0.42.json, deviceless_rtx3090_cudnn9.28.0.42_gpuhost.jsonl, *.console.log
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
IMAGE=${IMAGE:-cloudgpufit-bench:cu124-v1}
IID=$(docker image inspect --format '{{.Id}}' "$IMAGE")
VERS=${VERS:-"9.5.1.17 9.7.1.26 9.10.2.21 9.12.0.46 9.28.0.42"}
SIGS=${SIGS:-conv_signatures_all.json}
U="$(id -u):$(id -g)"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader | tee "$D/nvidia-smi_version_test.txt"

probe() {  # $1 label, $2 site dir ("" = image's own cuDNN)
  local env=()
  if [ -n "$2" ]; then L=/probe/$2/nvidia/cudnn/lib; env=(-e LD_LIBRARY_PATH=$L -e LD_PRELOAD=$L/libcudnn.so.9); fi
  docker run --rm --gpus all --network none --user "$U" -e CUDA_MODULE_LOADING=LAZY -e IMAGE_ID="$IID" \
    -e GPU_LABEL="$1" -e HOME=/tmp "${env[@]}" -v "$D":/probe -w /probe --entrypoint python "$IMAGE" \
    /probe/cudnn_ws_probe.py --signatures "/probe/$SIGS" --out "/probe/probe_$1.jsonl" 2>&1 | tee -a "$D/probe_$1.console.log"
  head -1 "$D/probe_$1.jsonl" | python3 -c "import json,sys;e=json.load(sys.stdin)['environment'];print('LOADED cuDNN',e['cudnn'],'torch',e['torch'])"
}
fetch() {  # $1 package==version -> ./wheels, sha256 checked against wheels_SHA256SUMS
  mkdir -p "$D/wheels"
  docker run --rm --user "$U" -e HOME=/tmp -v "$D":/probe -w /probe --entrypoint pip "$IMAGE" download -q --no-deps -d wheels "$1"
  (cd "$D/wheels" && grep -F "$(ls | grep -i "$(echo "${1%%==*}" | tr - _)-${1##*==}-")" ../wheels_SHA256SUMS | sha256sum -c -)
}
install() {  # $1 wheel glob, $2 target dir
  docker run --rm --network none --user "$U" -e HOME=/tmp -v "$D":/probe -w /probe --entrypoint bash "$IMAGE" \
    -c "pip install -q --no-deps --no-index --target /probe/$2 /probe/wheels/$1"
}

probe rtx3090_cudnn9.1.0 ""
for v in $VERS; do
  fetch "nvidia-cudnn-cu12==$v"
  install "nvidia_cudnn_cu12-$v-*.whl" "site_$v"
  probe "rtx3090_cudnn$v" "site_$v"
  rm -f "$D"/wheels/nvidia_cudnn_cu12-$v-*.whl
  [ "$v" = "9.28.0.42" ] || rm -rf "$D/site_$v"
done
# deviceless (cuDNN 9.28 + frontend 1.32.2)
fetch "nvidia-cudnn-frontend==1.32.2"
install "nvidia_cudnn_frontend-1.32.2-*.whl" "site_9.28.0.42"
# 2026-11-23 INUGPU proposal (accepted): cuDNN 9.28 needs libcublasLt and libcudart, which torch preloads in the probe
# but nothing loads in the deviceless run -> add the image's cuBLAS and CUDA runtime lib dirs
NV=/opt/conda/lib/python3.11/site-packages/nvidia
DL=(-e PYTHONPATH=/probe/site_9.28.0.42 -e LD_LIBRARY_PATH=/probe/site_9.28.0.42/nvidia/cudnn/lib:$NV/cublas/lib:$NV/cuda_runtime/lib)
docker run --rm --gpus all --network none --user "$U" -e HOME=/tmp "${DL[@]}" -v "$D":/probe -w /probe --entrypoint python "$IMAGE" \
  /probe/deviceless_query.py --props /probe/props_rtx3090_cudnn9.28.0.42.json --save-props-from-gpu 2>&1 | tee "$D/deviceless.console.log"
docker run --rm --gpus all --network none --user "$U" -e HOME=/tmp "${DL[@]}" -v "$D":/probe -w /probe --entrypoint python "$IMAGE" \
  /probe/deviceless_query.py --props /probe/props_rtx3090_cudnn9.28.0.42.json --signatures "/probe/$SIGS" \
  --out /probe/deviceless_rtx3090_cudnn9.28.0.42_gpuhost.jsonl 2>&1 | tee -a "$D/deviceless.console.log"
rm -rf "$D/wheels" "$D/site_9.28.0.42"
sha256sum "$D"/probe_rtx3090_cudnn*.jsonl "$D"/props_*.json "$D"/deviceless_*.jsonl | tee "$D/RESULTS_SHA256SUMS"
echo VERSION_TEST_DONE
