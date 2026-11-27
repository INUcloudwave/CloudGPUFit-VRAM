#!/bin/bash
# 2026-11-16: allocator traces for high-slack conditions (reserved composition analysis). Same worker/protocol (trace = load, warmup 3, measure 1).
cd /home/inugpu/cgf-fp16-vram-20261111
IDS=gpt2__b128__in128__new512,gpt2__b128__in512__new512,tinyllama-1.1b-chat__b128__in128__new512,tinyllama-1.1b-chat__b128__in128__new256,t5-base__b128__in128__new512,opt-125m__b16__in128__new512
echo TRACE_START $(date -Iseconds)
python3 -I src/run_conditions.py --conditions conditions_main395_l256.json --list $IDS --mode trace --gpu-label rtx3090__trace_slack --attn default || echo RUNNER_ABORT
echo TRACE_END $(date -Iseconds)
