#!/usr/bin/env bash
# One-time setup, generalisation-test LLMs (2026-11-23). Run from the bundle root:
#   bash cgf-fp16-680-20261111/aws/setup_newmodels.sh <image.tar.zst>
set -euo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd); E=$R/cgf-fp16-680-20261111
IMG=${1:?image tar.zst}
echo "== host"; nproc; free -g | head -2; df -h "$R" | tail -1
grep -o -m1 'avx512_fp16' /proc/cpuinfo >/dev/null && echo "CPU flag avx512_fp16: yes" || echo "CPU flag avx512_fp16: NO (fp16_real will be slow)"
grep -o -m1 'amx_fp16' /proc/cpuinfo >/dev/null && echo "CPU flag amx_fp16: yes" || echo "CPU flag amx_fp16: no"
sudo -n true 2>/dev/null && echo "sudo without password: yes (auto stop works)" || echo "WARN: sudo needs a password -> auto stop will NOT work"
command -v docker >/dev/null || { echo "install docker first"; exit 1; }
echo "== image sha256 (want 8642784f94181ff45045b04cc61fd772b4f70a387ef65ce1167c1532f8ef1897)"; sha256sum "$IMG"
echo "== load image"; zstd -dc "$IMG" | docker load
GOT=$(docker image inspect cloudgpufit-cpu-validation:20261108 --format '{{.Id}}'); echo "$GOT"
[ "$GOT" = "sha256:875290b5854ad6dbc8ea554fecb292dd5ffdfea2a03e8c4e8179a3c827968fa5" ] || { echo "IMAGE ID MISMATCH"; exit 1; }
echo "== download 3 models (network, ~2.6 GB)"; mkdir -p "$E/hf_cache"
NET=1 MEM=16g CPUS=8 bash "$E/aws/in_image.sh" -u aws/download_models.py /exp/aws/model_files_newmodels.json | tail -4
echo "== xMem lock check"
MEM=4g CPUS=1 bash "$E/aws/in_image.sh" -c "from cgf_diag.replay import verify_xmem; print(verify_xmem('/work/xMem','/exp/provenance/xmem-allocator-lock.json')['ok'])"
echo "== smoke test (2 small conditions, fp16_real)"
cd "$E" && python3 tools/run_aws.py --run-name aws_newmodels_smoke --cases manifest/cases_newmodels_llm_20261123.jsonl \
   --stats aws/inuos_case_stats_newmodels.json --ids qwen2.5-0.5b-instruct__b1__in128__new128 pythia-410m-deduped__b1__in128__new128 \
   --worker-args "--dtype-mode fp16_real --models /exp/manifest/models_newmodels_20261122.json" | tail -2
for c in qwen2.5-0.5b-instruct__b1__in128__new128 pythia-410m-deduped__b1__in128__new128; do
  python3 -c "import json,sys;r=json.load(open('runs/aws_newmodels_smoke/cases/$c/result.json'));print('$c',r['status'],r.get('dtype_mode'),r.get('cublas_workspace_position'))"
done
echo "SETUP_DONE"
