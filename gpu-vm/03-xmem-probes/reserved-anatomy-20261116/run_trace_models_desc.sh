#!/bin/bash
cd /work/cgf-reserved-anatomy-20261116
for p in descending xmem; do
for d in in/rtx3090__trace_models__*; do
  c=${d#in/rtx3090__trace_models__}
  grep -q "\"policy\": \"$p\", \"condition\": \"$c\"" out/trace_models_addr_policy.jsonl && continue
  timeout 2400 python replay_addr_policy.py /work/xMem $p $d/trace.json >> out/trace_models_addr_policy.jsonl 2>>out/trace_models_addr_policy.err || echo "{\"policy\": \"$p\", \"condition\": \"$c\", \"error\": \"rc=$?\"}" >> out/trace_models_addr_policy.jsonl
done; done
echo REPLAYS_DONE >> out/trace_models_addr_policy.err
