#!/usr/bin/env bash
cd /home/inuos/cgf-fp16-680-20261111
run(){ MEM=$1 CPUS=$3 docs/run_in_image.sh -u runs/confirm_cnn/cnn_attr.py $2 /exp/manifest/cases395_l256.jsonl /exp/runs/main395_l256/cases $3 /exp/runs/confirm_cnn/attr.jsonl 2>&1 | grep CNN_ATTR; }
for c in resnet-50__b8__r512 mobilenet-v2-1.0-224__b32__r224 resnet-50__b32__r256 resnet-18__b16__r512 mobilenet-v2-1.0-224__b128__r224 resnet-50__b128__r224; do run 20g $c 5; done
run 12g mobilenet-v2-1.0-224__b8__r512 1
run 12g mobilenet-v2-1.0-224__b8__r512 10
run 12g resnet-50__b8__r512 1
run 12g resnet-50__b8__r512 10
run 30g mobilenet-v2-1.0-224__b128__r512 5
run 30g resnet-50__b128__r512 5
echo CNN_CONFIRM_DONE
