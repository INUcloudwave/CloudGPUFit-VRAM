#!/usr/bin/env bash
set -u
set -o pipefail
cd /home/inuos/CloudGPUFit-cpu-singlepass-20261108 || exit 1
mapfile -t cgf_case_args < <(python3 - <<'PY'
import json
from pathlib import Path
source=Path('/home/inuos/cloudgpufit-predictions/parallel-all342-20261108-183250')
ids=[]
for path in sorted(source.glob('*.prediction.json')):
    d=json.loads(path.read_text())
    if d.get('status') != 'ok' and d.get('error_kind') == 'cgroup_oom':
        ids.append(d['case_id'])
assert len(ids)==16, f'Expected 16 final OOM cases, found {len(ids)}'
for cid in ids:
    print('--case-id')
    print(cid)
PY
)
[[ ${#cgf_case_args[@]} -eq 32 ]] || exit 1
# Set admission threshold above physical RAM so memory usage never delays a start.
cgf_total_gib=$(python3 -c 'import math; from pathlib import Path; print(math.ceil(int(next(l for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemTotal:")).split()[1])/1024**2))')
cgf_out="/home/inuos/cloudgpufit-predictions/oom-unlimited-manual-$(TZ=Asia/Seoul date +%Y%m%d-%H%M%S)"
python3 -u tools/run_prediction_recovery.py \
  --output-root "$cgf_out" \
  --snapshot-image cloudgpufit-cpu-validation:20261108 \
  "${cgf_case_args[@]}" \
  --jobs 1 --threads 4 \
  --memory-gib 0 --solo-memory-gib 0 \
  --kill-below-available-gib 0 \
  --host-reserve-gib 0 --start-below-used-gib "$cgf_total_gib" \
  --timeout-seconds 21600 \
  2>&1 | tee "${cgf_out}.console.log"
