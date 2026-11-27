#!/usr/bin/env bash
# L256 main run: 151 cases (150 new + 1 never attempted), P = 3 parallel slots with low-memory preemption of the
# most recently started case, then S = memory failures / timeouts sequentially. Re-running resumes.
set -uo pipefail
cd /home/inuos/cgf-fp16-680-20261111
mkdir -p results/main395_l256
python3 -u tools/run_l256.py --run-name main395_l256 --cases manifest/cases395_l256.jsonl --phases P S 2>&1 | tee -a runs/main395_l256.console.log
python3 tools/collect395.py --run-name main395_l256 2>&1 | tee -a runs/main395_l256.console.log
echo MAIN395_L256_DONE | tee -a runs/main395_l256.console.log
