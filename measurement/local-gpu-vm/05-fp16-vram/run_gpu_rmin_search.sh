#!/bin/bash
cd /home/inugpu/cgf-fp16-vram-20261111
echo START $(date -Iseconds)
python3 -I tools/gpu_rmin_search.py results/gpu_rmin_search.jsonl rtx3090__capsearch \
  runs/rtx3090__trace_slack/opt-125m__b16__in128__new512/trace.json \
  runs/rtx3090__trace_models/qwen2.5-1.5b-instruct__b128__in128__new512/trace.json \
  runs/rtx3090__trace_models/resnet-50__b128__r512/trace.json \
  runs/rtx3090__trace_slack/tinyllama-1.1b-chat__b128__in128__new256/trace.json \
  runs/rtx3090__trace_slack/gpt2__b128__in512__new512/trace.json \
  runs/rtx3090__trace_models/t5-small__b128__in128__new512/trace.json \
  runs/rtx3090__trace_slack/tinyllama-1.1b-chat__b128__in128__new512/trace.json
echo ALL_END $(date -Iseconds)
