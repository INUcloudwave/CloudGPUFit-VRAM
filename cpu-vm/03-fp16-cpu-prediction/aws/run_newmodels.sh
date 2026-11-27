#!/usr/bin/env bash
# Main run, generalisation-test LLMs (48 conditions, fp16_real), inside tmux:
#   tmux new -s cgf 'bash cgf-fp16-680-20261111/aws/run_newmodels.sh 2>&1 | tee cgf-fp16-680-20261111/runs/aws_newmodels.console.log'
# AUTO_STOP=1 (default): after AWS_RUN_DONE the OS shuts down in 2 min. With the EC2 shutdown behavior "stop"
# the instance is STOPPED (EBS disk and results kept), not terminated. AUTO_STOP=0 disables it.
set -uo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd); E=$R/cgf-fp16-680-20261111; cd "$E"
RUN=${RUN:-aws_fp16_newmodels}
CASES=manifest/cases_newmodels_llm_20261123.jsonl
echo "== $(date -Is) phase P+S run=$RUN"
python3 tools/run_aws.py --run-name $RUN --cases $CASES --stats aws/inuos_case_stats_newmodels.json \
   --worker-args "--dtype-mode fp16_real --models /exp/manifest/models_newmodels_20261122.json"
echo "== $(date -Is) reserved prediction v3 (descending + workspace, RTX 3090 capacity)"
NJ=$(( $(nproc) - 2 )); [ $NJ -gt 16 ] && NJ=16
MEM=$(( $(free -g | awk '/Mem:/{print $2}') - 16 ))g CPUS=$(nproc) bash aws/in_image.sh -u tools/reserved_predict.py \
   --name $RUN --gpu rtx3090 --jobs $NJ --runs-root runs/$RUN --cases $CASES --ws-source result
echo "== $(date -Is) pack results"
tar --exclude='*.trace.json' -czf "$R/${RUN}_results.tar.gz" runs/$RUN results/reserved_v3/$RUN aws/inuos_case_stats_newmodels.json
ls -la "$R/${RUN}_results.tar.gz"; sha256sum "$R/${RUN}_results.tar.gz"
echo "AWS_RUN_DONE"
if [ "${AUTO_STOP:-1}" = 1 ]; then
  sync
  echo "== $(date -Is) auto stop: shutdown in 2 min (cancel: sudo shutdown -c)"
  sudo -n shutdown -h +2 "cgf run done" || echo "WARN: sudo shutdown failed, instance keeps running"
fi
