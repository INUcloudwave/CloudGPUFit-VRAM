#!/usr/bin/env bash
cd /home/inuos/cgf-reserved-diag-20261110
D=runs/diag_oom
for cfg in "none disabled" "memory disabled" "memory enabled" "nomemory disabled"; do
  set -- $cfg; tag=prof_$1__gc_$2
  docs/run_in_image.sh -u -m cgf_diag.diag_oom --input-json inputs/cpu_production/core__decoder_llm__medium__fp32__12.input.json \
     --profile $1 --gc $2 --n 3 --out /exp/$D/$tag.jsonl > $D/$tag.log 2>&1
  echo "$tag exit=$?" >> $D/matrix_status.txt
done
echo MATRIX_DONE >> $D/matrix_status.txt
