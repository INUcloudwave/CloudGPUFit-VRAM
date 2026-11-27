#!/usr/bin/env bash
# pilot v2: transformers default attention (sdpa where supported), FP16. eager pilot (runs/rtx3090) is kept as is.
cd /home/inugpu/cgf-fp16-vram-20261111
L=rtx3090__attn-default
echo "V2_START $(date -Is)"
python3 -I src/run_conditions.py --list pilot --mode measure --gpu-label $L --attn default || echo "MEASURE_RUNNER_FAILED"
echo "MEASURE_END $(date -Is)"
python3 -I src/run_conditions.py --list pilot --mode validate --gpu-label $L --attn default || echo "VALIDATE_RUNNER_FAILED"
echo "VALIDATE_END $(date -Is)"
python3 -I src/run_conditions.py --list config/trace_pilot_v2.json --mode trace --gpu-label $L --attn default || echo "TRACE_RUNNER_FAILED"
python3 -I src/run_conditions.py --list config/trace_pilot_v2.json --mode trace --gpu-label rtx3090 --attn eager || echo "TRACE_EAGER_RUNNER_FAILED"
echo "TRACE_END $(date -Is)"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" && docker run --rm --pull=never --network none --gpus device=GPU-17eb7299-8b39-914d-7c5b-26908502904f --user $(id -u):$(id -g) -e HOME=/tmp -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e CUDA_MODULE_LOADING=LAZY -v /home/inugpu/cloudgpufit-bench/hf-cache:/hf:ro -v $PWD/tools:/tools:ro -v $PWD/config:/config:ro -v $PWD/results:/out -w /tmp --entrypoint python cloudgpufit-bench:cu124-v1 -I /tools/diag_attention_kernels.py /config /out/attention_kernels_rtx3090_default.json opt-125m__b8__in128__new128 opt-125m__b16__in512__new512 tinyllama-1.1b-chat__b8__in128__new128 tinyllama-1.1b-chat__b16__in512__new512 qwen2.5-1.5b-instruct__b8__in128__new128 qwen2.5-1.5b-instruct__b16__in512__new512 gpt2__b8__in128__new128 gpt2__b16__in512__new512 gpt2-medium__b8__in128__new128 gpt2-medium__b16__in512__new512 gpt2-xl__b8__in128__new128 gpt2-xl__b16__in512__new512 vit-large-patch16-224__b8__r224 vit-large-patch16-224__b16__r224 > results/attention_kernels_rtx3090_default.log 2>&1 || echo "KERNEL_DIAG_FAILED"
echo "V2_DONE $(date -Is)"; df -h / | tail -1
