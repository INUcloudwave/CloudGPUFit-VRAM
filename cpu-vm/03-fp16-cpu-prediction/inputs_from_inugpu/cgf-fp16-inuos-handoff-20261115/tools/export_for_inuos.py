#!/usr/bin/env python3
"""Flatten runs/<label>/<cid>/{measure,validate}.json into one per-condition CSV (exact bytes, full input sha256)
for the INUOS CPU-prediction comparison.  python3 -I tools/export_for_inuos.py runs/<label> <conditions.json> <out.csv>
Missing result files are written as status=missing (run still in progress)."""
import csv, json, sys
from pathlib import Path

base, conds, out = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())["conditions"], Path(sys.argv[3])
COLS = ["condition_id", "model", "workload_type", "batch_size", "input_len", "max_new_tokens", "resolution",
        "measure_status", "measure_failed_stage", "attn_implementation", "attn_requested", "class", "param_bytes",
        "cuda_total_memory", "after_load_reserved", "after_load_allocated",
        "m1_start_reserved", "m1_start_allocated", "m1_peak_reserved", "m1_peak_allocated",
        "m1_end_with_output_reserved", "m1_end_with_output_allocated",
        "median_peak_reserved", "max_peak_reserved", "median_peak_allocated", "max_peak_allocated",
        "max_nvml_process_peak", "measure_new_tokens", "input_sha256",
        "fail_reserved_peak", "fail_allocated_peak", "fail_reserved_current", "fail_allocated_current",
        "fail_num_alloc_retries", "fail_error_head",
        "validate_status", "validate_failed_stage", "validate_all_finite", "validate_steps_checked",
        "validate_first_nonfinite_step", "worker_sha256", "conditions_json_sha256"]


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


rows = []
for c in conds:
    cid = c["condition_id"]
    r = {k: c.get(k, "") for k in ("condition_id", "model", "workload_type", "batch_size", "input_len",
                                   "max_new_tokens", "resolution")}
    m, v = load(base / cid / "measure.json"), load(base / cid / "validate.json")
    if m is None:
        r["measure_status"] = "missing"
    else:
        r.update(measure_status=m["status"], measure_failed_stage=m.get("failed_stage") or "",
                 worker_sha256=m.get("worker_sha256"), conditions_json_sha256=m.get("conditions_json_sha256"),
                 cuda_total_memory=m.get("environment", {}).get("cuda_total_memory"))
        md = m.get("model") or {}
        r.update(attn_implementation=md.get("attn_implementation"), attn_requested=md.get("attn_requested"),
                 **{"class": md.get("class")}, param_bytes=md.get("param_bytes"))
        al = m.get("after_load") or {}
        r.update(after_load_reserved=al.get("reserved_bytes.all.current"),
                 after_load_allocated=al.get("allocated_bytes.all.current"))
        r["input_sha256"] = ";".join(f"{k}={x['sha256']}" for k, x in (m.get("inputs") or {}).items())
        s = m.get("summary") or {}
        r.update(m1_start_reserved=s.get("first_measure_start_reserved"),
                 m1_start_allocated=s.get("first_measure_start_allocated"),
                 m1_peak_reserved=s.get("first_measure_peak_reserved"),
                 m1_peak_allocated=s.get("first_measure_peak_allocated"),
                 median_peak_reserved=s.get("median_peak_reserved"), max_peak_reserved=s.get("max_peak_reserved"),
                 median_peak_allocated=s.get("median_peak_allocated"), max_peak_allocated=s.get("max_peak_allocated"),
                 max_nvml_process_peak=s.get("max_nvml_process_peak"))
        ms = m.get("measures") or []
        if ms:
            r.update(m1_end_with_output_reserved=ms[0]["end_with_output"]["reserved"],
                     m1_end_with_output_allocated=ms[0]["end_with_output"]["allocated"],
                     measure_new_tokens="|".join(str(x.get("new_tokens")) for x in ms))
        f = m.get("stats_at_failure")
        if f:
            r.update(fail_reserved_peak=f.get("reserved_bytes.all.peak"), fail_allocated_peak=f.get("allocated_bytes.all.peak"),
                     fail_reserved_current=f.get("reserved_bytes.all.current"),
                     fail_allocated_current=f.get("allocated_bytes.all.current"),
                     fail_num_alloc_retries=f.get("num_alloc_retries"), fail_error_head=(m.get("error") or "")[:160])
    if v is None:
        r["validate_status"] = "missing"
    else:
        val = v.get("validation") or {}
        r.update(validate_status=v["status"], validate_failed_stage=v.get("failed_stage") or "",
                 validate_all_finite=val.get("all_finite"), validate_steps_checked=val.get("steps_checked"),
                 validate_first_nonfinite_step=val.get("first_nonfinite_step"))
    rows.append(r)
with out.open("w", newline="") as fh:
    w = csv.DictWriter(fh, COLS); w.writeheader(); w.writerows(rows)
from collections import Counter
print(out, len(rows), dict(Counter(r["measure_status"] for r in rows)), dict(Counter(r["validate_status"] for r in rows)))
