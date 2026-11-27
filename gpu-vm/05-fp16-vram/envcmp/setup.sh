#!/bin/bash
# One-time setup on the g4dn.xlarge: checksums -> preflight -> load + verify both images -> model cache (download at
# pinned revisions if missing) + full hash verify -> runner dry run (must select 39). Idempotent.
#   bash envcmp/setup.sh <HF_CACHE_DIR>
set -uo pipefail
EXP=$(readlink -f "$(dirname "$0")/.."); TOP=$(dirname "$EXP"); cd "$EXP"; CACHE=$(readlink -f "${1:?hf cache dir}"); mkdir -p "$CACHE" logs
{
echo "SETUP_START $(date -Iseconds)"
(cd "$TOP" && sha256sum -c SHA256SUMS) || { echo "ABORT top checksum"; exit 1; }
sha256sum -c --quiet MANIFEST_SHA256SUMS || { echo "ABORT bundle checksum"; exit 1; }
bash aws/aws_00_preflight.sh g4dn.xlarge || { echo "ABORT preflight (expected g4dn.xlarge, 1 GPU)"; exit 1; }
if ! python3 -I envcmp/verify_images.py aws/IMAGES_INFO_envcmp.json; then
  echo "loading images from $TOP/image/cgf-envcmp-images.docker-save.tar"
  docker load -i "$TOP/image/cgf-envcmp-images.docker-save.tar" && python3 -I envcmp/verify_images.py aws/IMAGES_INFO_envcmp.json \
    || { echo "ABORT image verify"; exit 1; }
fi
bash aws/aws_02_cache.sh verify "$CACHE" || { bash aws/aws_02_cache.sh download "$CACHE" || { echo "ABORT model cache"; exit 1; }; }
python3 -I tools/run_conditions_env.py --conditions conditions_envcmp39.json --list all --mode measure \
  --gpu-label dryrun --attn default --hf-cache "$CACHE" --image cloudgpufit-bench:cu124-v1 --env-name A --dry-run | tail -1
echo "SETUP_END $(date -Iseconds)"
} 2>&1 | tee -a logs/envcmp_setup.txt
