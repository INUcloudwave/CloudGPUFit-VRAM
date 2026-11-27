#!/bin/bash
# 2026-11-17: local smoke test of the env-comparison wrapper/runner and the C image on RTX 3090 (waits for the cu128 run).
cd /home/inugpu/cgf-fp16-vram-20261111
while ! grep -q "CU128_END\|RUNNER_ABORT\|STOP" logs/cu128_rtx3090.txt; do sleep 60; done
while pgrep -f run_cu128_rtx3090.sh >/dev/null; do sleep 30; done
echo SMOKE_START $(date -Iseconds)
IDS=opt-125m__b1__in128__new128,resnet-50__b16__r512,tinyllama-1.1b-chat__b1__in128__new128
for E in A C; do
  IMG=cloudgpufit-bench:cu124-v1; [ $E = C ] && IMG=cloudgpufit-bench:pt251-cu124-v1
  python3 -I tools/run_conditions_env.py --conditions conditions_main395_l256.json --list $IDS --mode measure \
    --gpu-label rtx3090__envcmp_smoke_$E --attn default --image $IMG --env-name $E || echo RUNNER_ABORT $E
  mkdir -p runs/rtx3090__envcmp_smoke_$E/_diag
  docker run --rm --pull=never --network none --gpus all --user $(id -u):$(id -g) -e HOME=/tmp -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
    -e CUDA_MODULE_LOADING=LAZY -v /home/inugpu/cloudgpufit-bench/hf-cache:/hf:ro -v $PWD/tools:/tools:ro -v $PWD/config_diag_smoke:/config:ro \
    -v $PWD/runs/rtx3090__envcmp_smoke_$E/_diag:/out -w /tmp --entrypoint python $IMG -I /tools/diag_attention_kernels.py /config \
    /out/attention_kernels.json opt-125m__b1__in128__new128 tinyllama-1.1b-chat__b1__in128__new128 vit-large-patch16-224__b1__r224 \
    > runs/rtx3090__envcmp_smoke_$E/_diag/attention_kernels.log 2>&1 || echo KERNEL_DIAG_FAILED $E
done
echo '{"condition_ids":["opt-125m__b1__in128__new128","resnet-50__b16__r512","tinyllama-1.1b-chat__b1__in128__new128"]}' > /tmp/claude-1000/-home-inugpu/fae1c204-5bd8-46d5-955c-73f894275b96/scratchpad/smoke_ids.json
python3 -I tools/compare_envs.py . /tmp/claude-1000/-home-inugpu/fae1c204-5bd8-46d5-955c-73f894275b96/scratchpad/smoke_ids.json results/envcmp_smoke_3090 \
  A=runs/rtx3090__envcmp_smoke_A C=runs/rtx3090__envcmp_smoke_C REF=runs/rtx3090__main395_l256
echo SMOKE_END $(date -Iseconds)
