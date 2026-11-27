#!/usr/bin/env python3
"""Reuse finished CPU results of conditions shared by two condition tables (no re-run).

For every condition_id present in both tables whose condition definition (model, workload, batch, L, N / R) is
identical and whose source case dir has a final result.json, create
    runs/<dst>/cases/<cid>  ->  ../../<src>/cases/<cid>   (relative symlink, source untouched)
Final = any status except running/None. Failures (e.g. worker_killed_cpu_oom) are linked too: they are NOT re-run.
Writes runs/<dst>/reuse_manifest.json. Idempotent; never overwrites an existing dst entry.

  python3 tools/link_overlap.py --src main395 --src-cases manifest/cases395.jsonl \
      --dst main395_l256 --dst-cases manifest/cases395_l256.jsonl
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

EXP = Path("/home/inuos/cgf-fp16-680-20261111")
KEYS = ("model", "kind", "dtype", "batch_size", "input_len", "new_tokens", "resolution")


def load(p):
    return {c["case_id"]: c for c in map(json.loads, (EXP / p).read_text().splitlines())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--src-cases", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--dst-cases", required=True)
    a = ap.parse_args()
    src, dst = load(a.src_cases), load(a.dst_cases)
    src_root, dst_root = EXP / "runs" / a.src, EXP / "runs" / a.dst
    (dst_root / "cases").mkdir(parents=True, exist_ok=True)
    linked, skipped = [], []
    for cid in sorted(set(src) & set(dst)):
        if any(src[cid].get(k) != dst[cid].get(k) for k in KEYS):
            skipped.append({"condition_id": cid, "reason": "definition differs"})
            continue
        sdir = src_root / "cases" / cid
        rp = sdir / "result.json"
        if not rp.exists():
            skipped.append({"condition_id": cid, "reason": "no result in source (will be run)"})
            continue
        r = json.loads(rp.read_text())
        if r.get("status") in (None, "running"):
            skipped.append({"condition_id": cid, "reason": f"source status {r.get('status')}"})
            continue
        d = dst_root / "cases" / cid
        if not d.exists() and not d.is_symlink():
            os.symlink(os.path.relpath(sdir, d.parent), d)
        linked.append({"condition_id": cid, "status": r.get("status"), "source": str(sdir.relative_to(EXP)),
                       "result_sha256": hashlib.sha256(rp.read_bytes()).hexdigest()})
    man = {"created": time.strftime("%FT%T%z"), "src_run": a.src, "src_cases": a.src_cases, "dst_cases": a.dst_cases,
           "dst_cases_sha256": hashlib.sha256((EXP / a.dst_cases).read_bytes()).hexdigest(),
           "definition_keys": KEYS, "linked": linked, "skipped": skipped,
           "counts": {"overlap": len(set(src) & set(dst)), "linked": len(linked), "skipped": len(skipped),
                      "to_run": len(set(dst) - {x["condition_id"] for x in linked})}}
    (dst_root / "reuse_manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man["counts"]), json.dumps(skipped))


if __name__ == "__main__":
    main()
