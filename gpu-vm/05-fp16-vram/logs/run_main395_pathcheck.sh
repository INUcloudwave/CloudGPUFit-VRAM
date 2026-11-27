#!/bin/bash
# Path check of the updated runner (--conditions conditions_main395.json, attn default) on 3 representative conditions.
cd /home/inugpu/cgf-fp16-vram-20261111
L=gpt2__b8__in128__new128,resnet-18__b8__r224,opt-125m__b1__in128__new128
echo PATHCHECK_START $(date -Iseconds)
for M in measure validate; do
  python3 -I src/run_conditions.py --conditions conditions_main395.json --list $L --mode $M --gpu-label rtx3090__main395-pathcheck --attn default
done
echo PATHCHECK_DONE $(date -Iseconds)
