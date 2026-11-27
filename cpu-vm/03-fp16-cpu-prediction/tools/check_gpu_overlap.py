#!/usr/bin/env python3
"""GPU L256 vs GPU L1024 on the 245 shared condition_ids (same worker/config -> values should be identical).
  python3 tools/check_gpu_overlap.py OLD_L1024.csv NEW_L256.csv [--out results/main395_l256/gpu_overlap_check.json]"""
import argparse
import csv
import json

COLS = ["measure_status", "measure_failed_stage", "attn_implementation", "after_load_reserved", "after_load_allocated",
        "m1_start_reserved", "m1_start_allocated", "m1_peak_reserved", "m1_peak_allocated", "input_sha256", "worker_sha256"]

ap = argparse.ArgumentParser()
ap.add_argument("old")
ap.add_argument("new")
ap.add_argument("--out", default=None)
a = ap.parse_args()
old = {r["condition_id"]: r for r in csv.DictReader(open(a.old))}
new = {r["condition_id"]: r for r in csv.DictReader(open(a.new))}
shared = sorted(set(old) & set(new))
diffs = []
for cid in shared:
    d = {c: (old[cid].get(c), new[cid].get(c)) for c in COLS if old[cid].get(c) != new[cid].get(c)}
    if d:
        diffs.append({"condition_id": cid, **d})
res = {"shared": len(shared), "identical": len(shared) - len(diffs), "different": len(diffs), "diffs": diffs}
print(json.dumps({k: v for k, v in res.items() if k != "diffs"}))
for d in diffs[:20]:
    print(json.dumps(d))
if a.out:
    open(a.out, "w").write(json.dumps(res, indent=1))
