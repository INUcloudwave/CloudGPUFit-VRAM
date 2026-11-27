#!/usr/bin/env bash
# sequential main collection; stops at first failure (failed run dirs are kept)
R=/home/inugpu/cgf-reserved-diag-20261110; cd $R
CASES="core__decoder_llm__small__fp16__01 core__decoder_llm__small__fp16__12 core__decoder_llm__medium__fp32__12"
RUNS="R0_aws_replica R1_controlled R2_cold_single D1_cold_single_empty_cache"
echo "ALL_START $(date -Is)"
for c in $CASES; do for r in $RUNS; do
  ./run_one.sh $c $r || { echo "STOPPED at $c $r $(date -Is)"; exit 1; }
done; done
./run_one.sh core__decoder_llm__medium__fp32__12 R1_controlled all __stacks_all || { echo "OPTIONAL_FAILED $(date -Is)"; exit 2; }
echo "ALL_DONE $(date -Is)"; df -h / | tail -1
