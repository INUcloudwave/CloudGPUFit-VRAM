#!/bin/bash
cd /work/cgf-reserved-anatomy-20261116
for c in rtx3090__trace_slack__opt-125m__b16__in128__new512 rtx3090__trace_models__qwen2.5-1.5b-instruct__b128__in128__new512 rtx3090__trace_models__resnet-50__b128__r512 rtx3090__trace_slack__tinyllama-1.1b-chat__b128__in128__new256 rtx3090__trace_slack__gpt2__b128__in512__new512 rtx3090__trace_models__t5-small__b128__in128__new512 rtx3090__trace_slack__tinyllama-1.1b-chat__b128__in128__new512; do
  [ -s out/capsweep_desc__$c.json ] && continue
  timeout 10800 python capacity_sweep_desc.py /work/xMem in/$c/trace.json out/capsweep_desc__$c.json > out/capsweep_desc__$c.log 2>&1 || echo "FAIL $c rc=$?" >> out/capsweep_desc.err
done
echo SWEEP_DONE >> out/capsweep_desc.err
