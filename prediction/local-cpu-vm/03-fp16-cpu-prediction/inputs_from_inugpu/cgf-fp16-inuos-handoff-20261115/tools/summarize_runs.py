#!/usr/bin/env python3
"""Summarize measure/validate results of one GPU label into CSV + markdown (observed values only).

  python3 -I summarize_runs.py <root>/runs/<gpu_label> [condition_list.json] > summary.md
"""
import csv
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
ids = json.loads(Path(sys.argv[2]).read_text())["condition_ids"] if len(sys.argv) > 2 else sorted(
    p.name for p in base.iterdir() if p.is_dir())
M = 1 << 20
rows = []
for cid in ids:
    d = base / cid
    r = json.loads((d / "measure.json").read_text()) if (d / "measure.json").exists() else {}
    v = json.loads((d / "validate.json").read_text()) if (d / "validate.json").exists() else {}
    ln = json.loads((d / "measure.launch.json").read_text()) if (d / "measure.launch.json").exists() else {}
    s = r.get("summary") or {}
    ms = r.get("measures") or []
    rows.append({
        "condition_id": cid, "status": r.get("status", "not_run"), "failed_stage": r.get("failed_stage"),
        "error": (r.get("error") or "")[:160],
        "load_reserved_MiB": (r.get("after_load") or {}).get("reserved_bytes.all.current", 0) / M if r.get("after_load") else None,
        "m1_start_reserved_MiB": s.get("first_measure_start_reserved", 0) / M if s else None,
        "m1_peak_reserved_MiB": s.get("first_measure_peak_reserved", 0) / M if s else None,
        "m1_peak_allocated_MiB": s.get("first_measure_peak_allocated", 0) / M if s else None,
        "median_peak_reserved_MiB": s.get("median_peak_reserved", 0) / M if s else None,
        "max_peak_reserved_MiB": s.get("max_peak_reserved", 0) / M if s else None,
        "max_nvml_peak_MiB": (s.get("max_nvml_process_peak") or 0) / M if s.get("max_nvml_process_peak") else None,
        "new_tokens": sorted({m.get("new_tokens") for m in ms}) if ms else None,
        "measure_s_median": sorted(m["elapsed_s"] for m in ms)[len(ms) // 2] if ms else None,
        "model_load_s": (r.get("timings_s") or {}).get("model_load"),
        "wall_s": ln.get("wall_s"),
        "validate_status": v.get("status", "not_run"),
        "validate_all_finite": (v.get("validation") or {}).get("all_finite"),
        "validate_steps": (v.get("validation") or {}).get("steps_checked"),
        "validate_first_nonfinite_step": (v.get("validation") or {}).get("first_nonfinite_step"),
    })
out = base / "summary.csv"
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0]))
    w.writeheader()
    w.writerows(rows)


def fmt(x):
    return "" if x is None else f"{x:.1f}" if isinstance(x, float) else str(x)


cols = ["condition_id", "status", "failed_stage", "load_reserved_MiB", "m1_start_reserved_MiB", "m1_peak_reserved_MiB",
        "max_peak_reserved_MiB", "m1_peak_allocated_MiB", "max_nvml_peak_MiB", "new_tokens", "wall_s",
        "validate_status", "validate_first_nonfinite_step"]
print("| " + " | ".join(cols) + " |")
print("|" + "---|" * len(cols))
for r in rows:
    print("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
print(f"\nwritten {out}", file=sys.stderr)
