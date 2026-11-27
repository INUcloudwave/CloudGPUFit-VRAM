#!/usr/bin/env python3
"""Binary search of the real GPU R_min (smallest allocator cap with status ok) per condition, 32 MiB resolution.

  python3 -I tools/gpu_rmin_search.py <out.jsonl> <label> <trace.json>...

For each condition (taken from an existing trace run): lo = ceil(peak allocated) - 1 (cap below peak allocated
cannot succeed, not run), hi = uncapped peak reserved + 2 (run first, must be ok). Each probe = one fresh container
run of tools/run_conditions_cap.py in measure mode (warmup 3 + measure 5, unchanged worker)."""
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
out_jsonl, label, traces = sys.argv[1], sys.argv[2], sys.argv[3:]
MiB = 2 ** 20
RES = 32


def probe(cid, cap):
    d = ROOT / "runs" / label / f"{cid}__cap{cap}"
    if not (d / "measure.json").exists():
        subprocess.run([sys.executable, "-I", str(ROOT / "tools/run_conditions_cap.py"), "--conditions",
                        "conditions_main395_l256.json", "--list", cid, "--mode", "measure", "--gpu-label", label,
                        "--mem-cap-mib", str(cap)], check=False)
    R = json.loads((d / "measure.json").read_text())
    s = R.get("summary") or {}
    retries = [w["end_with_output"]["num_alloc_retries"] for w in R.get("warmup", [])] + \
              [m.get("end_with_output", {}).get("num_alloc_retries") for m in R.get("measures", [])][:1]
    row = {"condition_id": cid, "cap_MiB": cap, "status": R["status"],
           "first_measure_peak_reserved_MiB": s.get("first_measure_peak_reserved", 0) / MiB or None,
           "first_measure_peak_allocated_MiB": round(s.get("first_measure_peak_allocated", 0) / MiB, 1) or None,
           "num_alloc_retries_warmup_then_m0": retries}
    with open(out_jsonl, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    return R["status"]


for tj in traces:
    T = json.loads(Path(tj).read_text())
    cid = T["condition"]["condition_id"]
    pk = T["measures"][0]["peak"]
    lo = math.ceil(pk["allocated"] / MiB) - 1
    hi = int(pk["reserved"] / MiB) + 2
    st = probe(cid, hi)
    if st != "ok":
        print(json.dumps({"condition_id": cid, "abort": f"hi={hi} status={st}"}), flush=True)
        continue
    bad = None
    while hi - lo > RES:
        mid = (lo + hi) // 2
        st = probe(cid, mid)
        if st == "ok":
            hi = mid
        elif st == "gpu_oom":
            lo = mid
        else:
            bad = st
            break
    res = {"condition_id": cid, "gpu_R_min_MiB_upper": hi, "gpu_fail_cap_MiB": lo, "resolution_MiB": RES,
           "uncapped_peak_reserved_MiB": pk["reserved"] / MiB, "peak_allocated_MiB": round(pk["allocated"] / MiB, 1)}
    if bad:
        res["aborted_on_status"] = bad
    with open(out_jsonl.replace(".jsonl", "_summary.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")
    print("RESULT", json.dumps(res), flush=True)
print("SEARCH_DONE", flush=True)
