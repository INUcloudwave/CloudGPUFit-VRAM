#!/bin/bash
# 2026-11-16: (A) allocator traces, one large condition per remaining model (reference traces for INUOS request-stream matching)
#             (B) run-to-run variability: same condition, fresh process, 15 repeats (measure mode; one label per repeat because the runner skips existing results)
# Same worker/runner/image as the main runs. Resumable: rerun this script and finished items are skipped.
cd /home/inugpu/cgf-fp16-vram-20261111
C=conditions_main395_l256.json
A=qwen2.5-1.5b-instruct__b128__in128__new512,qwen2.5-1.5b-instruct__b32__in512__new512,t5-3b__b16__in128__new512,t5-3b__b8__in128__new128,t5-small__b128__in128__new512,gpt2-medium__b128__in128__new512,gpt2-xl__b32__in128__new512,opt-125m__b128__in512__new512,tinyllama-1.1b-chat__b32__in512__new512,resnet-18__b128__r512,resnet-50__b128__r512,mobilenet-v2-1.0-224__b128__r512,vit-large-patch16-224__b128__r224
echo A_START $(date -Iseconds)
python3 -I src/run_conditions.py --conditions $C --list $A --mode trace --gpu-label rtx3090__trace_models --attn default || echo A_RUNNER_ABORT
echo A_END $(date -Iseconds)
B=opt-125m__b16__in128__new512,opt-125m__b32__in128__new512
echo B_START $(date -Iseconds)
for r in $(seq -w 1 15); do
  python3 -I src/run_conditions.py --conditions $C --list $B --mode measure --gpu-label rtx3090__repeat_r$r --attn default || echo B_RUNNER_ABORT r$r
done
echo B_END $(date -Iseconds)
echo ALL_END $(date -Iseconds)
