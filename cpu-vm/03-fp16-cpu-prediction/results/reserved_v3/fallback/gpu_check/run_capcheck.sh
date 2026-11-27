#!/bin/bash
cd /home/inugpu/cgf-fallback-check-20261123
echo START $(date -Iseconds)
for cap in 7946 5900 5640 5380 5130 4900 4872 4840; do
  python3 -I tools/run_conditions_cap.py --root /home/inugpu/cgf-fallback-check-20261123 --conditions conditions_newmodels_20261122.json --list regnet-y-040__b128__r512 --mode measure --gpu-label rtx3090__fallbackcheck --mem-cap-mib $cap --attn default --image cloudgpufit-bench:cu124-v1
done
echo CAPCHECK_DONE $(date -Iseconds)
