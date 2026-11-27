#!/usr/bin/env bash
# Main run (inside tmux):  tmux new -s cgf 'bash cgf-fp16-680-20261111/aws/run.sh 2>&1 | tee cgf-fp16-680-20261111/runs/aws_run.console.log'
#   MODE=fp16_real (default) | fp32_shadow_v2 (INUOS method + fixed size rule) ; RUN=<run name>
set -uo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd); E=$R/cgf-fp16-680-20261111; cd "$E"
MODE=${MODE:-fp16_real}
case $MODE in
  fp16_real) WA="--dtype-mode fp16_real"; RUN=${RUN:-aws_fp16_l256}; WS=result ;;
  fp32_shadow_v2) WA="--shadow-rule v2"; RUN=${RUN:-aws_shadowv2_l256}; WS="positions --ws-positions results/reserved_v3/gemm_exact_positions.jsonl" ;;
  *) echo "unknown MODE"; exit 1 ;;
esac
echo "== $(date -Is) phase P+S  mode=$MODE run=$RUN"
python3 tools/run_aws.py --run-name $RUN --worker-args "$WA"
echo "== $(date -Is) reserved prediction v3 (descending + workspace, R_inf/R_at/R_min)"
NJ=$(( $(nproc) - 2 )); [ $NJ -gt 48 ] && NJ=48
MEM=$(( $(free -g | awk '/Mem:/{print $2}') - 16 ))g CPUS=$(nproc) bash aws/in_image.sh -u tools/reserved_predict.py \
   --name $RUN --gpu rtx3090 --jobs $NJ --runs-root runs/$RUN --cases manifest/cases395_l256.jsonl --ws-source $WS
echo "== $(date -Is) pack results"
tar --exclude='*.trace.json' -czf "$R/${RUN}_results.tar.gz" runs/$RUN results/reserved_v3/$RUN aws/inuos_case_stats.json
ls -la "$R/${RUN}_results.tar.gz"
echo "AWS_RUN_DONE"
