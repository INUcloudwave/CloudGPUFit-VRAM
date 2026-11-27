#!/usr/bin/env python3
"""Join CPU predictions with GPU measurements and compute errors.

  python3 tools/compare_gpu.py --cpu results/main/cpu_results.jsonl --gpu <gpu_results.jsonl|csv> \
      --gpu-label rtx3090 --out results/main/compare_rtx3090.csv

Expected GPU columns (one row per case; names configurable with --map old=new):
  case_id, revision, status, input_<name>_sha256, protocol (warmup/measure tag),
  gpu_m1_allocated_peak_bytes, gpu_m1_reserved_peak_bytes  (FIRST measured invocation, compared)
  gpu_m1_allocated_start_bytes, gpu_m1_reserved_start_bytes (optional)
  gpu_max5_allocated_peak_bytes, gpu_max5_reserved_peak_bytes (max of 5 runs, capacity reference only)
A case is "comparable" only if case_id, revision and every input sha256 match, the GPU protocol tag
matches --protocol, and both sides have status ok. Others are listed with the reason and excluded from
numeric error aggregates (GPU OOM is reported as its own category).
"""
import argparse
import csv
import json
import statistics
from pathlib import Path


def load(path):
    p = Path(path)
    if p.suffix == ".jsonl":
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    if p.suffix == ".json":
        d = json.loads(p.read_text())
        return d if isinstance(d, list) else d.get("rows", [])
    with p.open() as f:
        return list(csv.DictReader(f))


def num(v):
    if v in (None, "", "None"):
        return None
    return int(float(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--gpu-label", required=True)
    ap.add_argument("--protocol", default="w3_m1_first_measured")
    ap.add_argument("--map", nargs="*", default=[], help="gpu_column=canonical_column")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ren = dict(x.split("=", 1) for x in a.map)
    cpu = {r["case_id"]: r for r in load(a.cpu)}
    gpu = {}
    for r in load(a.gpu):
        r = {ren.get(k, k): v for k, v in r.items()}
        gpu[r["case_id"]] = r
    rows = []
    for cid, c in cpu.items():
        g = gpu.get(cid)
        out = {"case_id": cid, "model": c["model"], "gpu_label": a.gpu_label, "cpu_status": c["status"],
               "gpu_status": g.get("status") if g else None}
        reasons = []
        if g is None:
            reasons.append("gpu_missing")
        else:
            if g.get("revision") and g["revision"] != c["revision"]:
                reasons.append("revision_mismatch")
            for k in [k for k in c if k.startswith("input_") and k.endswith("_sha256")]:
                if g.get(k) is None:
                    reasons.append(f"gpu_no_{k}")
                elif g[k] != c[k]:
                    reasons.append(f"{k}_mismatch")
            if g.get("protocol") not in (None, "", a.protocol):
                reasons.append("protocol_mismatch")
            gs = str(g.get("status", "")).lower()
            if "oom" in gs:
                reasons.append("gpu_oom")
            elif gs not in ("ok", "success"):
                reasons.append(f"gpu_status_{gs}")
        if c["status"] != "ok":
            reasons.append(f"cpu_{c['status']}")
        out["comparable"] = not reasons
        out["exclusion_reasons"] = ";".join(reasons)
        for m in ("allocated", "reserved"):
            cp = num(c.get(f"cpu_{m}_peak_bytes"))
            gp = num(g.get(f"gpu_m1_{m}_peak_bytes")) if g else None
            out[f"cpu_{m}_peak_bytes"], out[f"gpu_m1_{m}_peak_bytes"] = cp, gp
            out[f"cpu_{m}_start_bytes"] = num(c.get(f"cpu_{m}_start_bytes"))
            out[f"gpu_m1_{m}_start_bytes"] = num(g.get(f"gpu_m1_{m}_start_bytes")) if g else None
            out[f"gpu_max5_{m}_peak_bytes"] = num(g.get(f"gpu_max5_{m}_peak_bytes")) if g else None
            if cp is not None and gp:
                out[f"{m}_diff_bytes"] = cp - gp
                out[f"{m}_rel_err"] = (cp - gp) / gp
        rows.append(out)
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    comp = [r for r in rows if r["comparable"]]
    summ = {"gpu_label": a.gpu_label, "cases": len(rows), "comparable": len(comp),
            "excluded_by_reason": {}}
    for r in rows:
        for x in filter(None, r["exclusion_reasons"].split(";")):
            summ["excluded_by_reason"][x] = summ["excluded_by_reason"].get(x, 0) + 1
    for m in ("allocated", "reserved"):
        e = [abs(r[f"{m}_rel_err"]) for r in comp if r.get(f"{m}_rel_err") is not None]
        if e:
            summ[m] = {"n": len(e), "mean_abs_rel_err": statistics.mean(e), "median_abs_rel_err": statistics.median(e),
                       "max_abs_rel_err": max(e), "within_5pct": sum(x <= 0.05 for x in e),
                       "within_10pct": sum(x <= 0.10 for x in e)}
    Path(a.out).with_suffix(".summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
