#!/usr/bin/env bash
# After the pilot finishes: rerun gpt2 / gpt2-medium pilot B8 cases WITHOUT the Conv1D col-major change
# and check the request trace is identical to the pilot (col-major) run.
cd /home/inuos/cgf-fp16-680-20261111
while tmux has-session -t cgf680-pilot 2>/dev/null; do sleep 60; done
python3 -u tools/run_suite.py --run-name verify-layout-rowmajor --select ids \
  --ids fp16__gpt2__b8__in128__new128 fp16__gpt2-medium__b8__in128__new128 --threads 20 --mem 42g --timeout-hours 12 \
  2>&1 | tee -a runs/verify-layout.console.log
for c in fp16__gpt2__b8__in128__new128 fp16__gpt2-medium__b8__in128__new128; do
  docs/run_in_image.sh tools/compare_events.py runs/pilot/cases/$c runs/verify-layout-rowmajor/cases/$c | tee -a runs/verify-layout.console.log
done
