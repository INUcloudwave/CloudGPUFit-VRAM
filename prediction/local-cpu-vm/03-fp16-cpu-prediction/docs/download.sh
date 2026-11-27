#!/usr/bin/env bash
# Usage: docs/download.sh [model ...]   (no args = all pinned models)
E=/home/inuos/cgf-fp16-680-20261111
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/exp/hf_cache -v $E:/exp --entrypoint python \
  cloudgpufit-cpu-validation:20261108 /exp/docs/download_models.py /exp/manifest/model_pins.json "$@"
