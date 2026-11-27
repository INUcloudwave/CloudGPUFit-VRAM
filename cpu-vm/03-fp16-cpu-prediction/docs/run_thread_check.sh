#!/usr/bin/env bash
# Does the CPU request trace depend on torch threads? Same worker v2 / same launcher (run_l256, one slot at a time),
# threads 1 / 5 (old phase A) / 7 (new parallel slots) / 20 (old phase B and phase S). Reference = 20 threads.
# Output: runs/thread_check/t<k>/cases/<cid>/, results/thread_check/compare.jsonl
set -uo pipefail
cd /home/inuos/cgf-fp16-680-20261111
IDS="resnet-18__b8__r224 mobilenet-v2-1.0-224__b8__r224 gpt2__b1__in128__new128 t5-small__b1__in128__new128"
mkdir -p results/thread_check
for k in 20 7 5 1; do
  python3 -u tools/run_l256.py --run-name thread_check/t$k --cases manifest/cases395_l256.jsonl --ids $IDS \
      --cpusets 0-19 --par-threads $k --phases P 2>&1 | tee -a results/thread_check/run.log
done
: > results/thread_check/compare.jsonl
for id in $IDS; do
  docs/run_in_image.sh tools/compare_requests_npz.py /exp/runs/thread_check/t20/cases/$id /exp/runs/thread_check/t7/cases/$id \
      /exp/runs/thread_check/t5/cases/$id /exp/runs/thread_check/t1/cases/$id | tee -a results/thread_check/compare.jsonl
done
echo THREAD_CHECK_DONE
