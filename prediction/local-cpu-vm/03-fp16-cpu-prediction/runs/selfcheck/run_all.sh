#!/usr/bin/env bash
# FP16 self-check (real FP16 vs shadow v1 / v2) for one small variant per model; 4 parallel lanes.
cd /home/inuos/cgf-fp16-680-20261111
one(){ MEM=$2 CPUS=4 docs/run_in_image.sh -u runs/selfcheck/selfcheck_shadow.py $1 /exp/manifest/cases395_l256.jsonl $3 $4 $5 4 /exp/runs/selfcheck/selfcheck.jsonl 2>&1 | grep -E "SELFCHECK|Error|error" | cut -c1-300; }
lane1(){ one t5-3b__b1__in128__new128 30g 2 64 4; one opt-125m__b1__in128__new128 8g 2 64 4; one resnet-18__b1__r128 8g 2 224 0; }
lane2(){ one gpt2-xl__b1__in128__new128 20g 2 64 4; one gpt2__b1__in128__new128 8g 2 64 4; one resnet-50__b1__r128 8g 2 224 0; }
lane3(){ one qwen2.5-1.5b-instruct__b1__in128__new128 20g 2 64 4; one gpt2-medium__b1__in128__new128 8g 2 64 4; one mobilenet-v2-1.0-224__b1__r128 8g 2 224 0; }
lane4(){ one tinyllama-1.1b-chat__b1__in128__new128 16g 2 64 4; one t5-base__b1__in128__new128 8g 2 64 4; one vit-large-patch16-224__b1__r224 8g 2 224 0; }
lane1 & lane2 & lane3 & lane4 & wait
echo SELFCHECK_DONE
