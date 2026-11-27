#!/usr/bin/env bash
# Install a variant's packages into a host dir (no image build; disk-saving). Usage: install_variant.sh S2
set -euo pipefail
V=$1; LAB=$(cd "$(dirname "$0")/.." && pwd)
read -r BASE TORCH INDEX DIR < <(python3 -I -c "import json,sys;v=json.load(open('$LAB/images/variants.json'))['$V'];print(v['base_image'],v['torch'],v['index'],v['pkgdir'])")
[ -e "$DIR" ] && { echo "exists: $DIR"; exit 1; }
mkdir -p "$DIR"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$DIR":/opt/pkgs "$BASE" sh -c "
mkdir -p /opt/pkgs/.tmp && TMPDIR=/opt/pkgs/.tmp python -m pip install --no-cache-dir --target /opt/pkgs --index-url $INDEX --extra-index-url https://pypi.org/simple torch==$TORCH numpy==2.2.2 transformers==4.52.4 accelerate==1.4.0 nvidia-ml-py==12.570.86 Pillow==11.1.0 safetensors==0.8.0 tokenizers==0.21.4 &&
PYTHONPATH=/opt/pkgs python -m pip freeze --path /opt/pkgs > /opt/pkgs/environment.lock.txt && rm -rf /opt/pkgs/.tmp"
du -sh "$DIR"
