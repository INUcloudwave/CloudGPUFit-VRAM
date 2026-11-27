#!/bin/bash
cd /work/cgf-reserved-anatomy-20261116
for d in in/rtx3090__trace_models__*; do
  c=$(basename $d)
  for p in descending xmem; do
    timeout 1800 python replay_addr_policy.py /work/xMem $p $d/trace.json >> out/trace_models_addr_policy.jsonl 2>>out/trace_models_addr_policy.err || echo "{\"policy\":\"$p\",\"condition\":\"$c\",\"error\":\"rc=$?\"}" >> out/trace_models_addr_policy.jsonl
  done
  timeout 1800 python xmem_replay_variants.py /work/xMem $d/trace.json out/$c.json > out/$c.log 2>&1 || echo "VARIANTS_FAIL $c" >> out/trace_models_addr_policy.err
done
echo REPLAYS_DONE >> out/trace_models_addr_policy.err
