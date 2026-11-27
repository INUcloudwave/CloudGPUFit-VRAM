#!/usr/bin/env python3
"""Build combined CSV + data tables for the eager vs default-attention RTX 3090 pilot report (observed values only).

  python3 -I build_report.py <exp_root>  -> results/pilot_rtx3090_eager_vs_default.csv, prints markdown tables
"""
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
M = 1 << 20
pilot = json.loads((root / "config/pilot.json").read_text())["condition_ids"]
LABELS = {"eager": root / "runs/rtx3090", "default": root / "runs/rtx3090__attn-default"}


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def mib(x):
    return None if x is None else round(x / M, 1)


rows = []
for cid in pilot:
    row = {"condition_id": cid}
    for tag, base in LABELS.items():
        r = load(base / cid / "measure.json") or {}
        v = load(base / cid / "validate.json") or {}
        s = r.get("summary") or {}
        ms = r.get("measures") or [{}]
        row.update({
            f"{tag}_attn_impl": (r.get("model") or {}).get("attn_implementation"),
            f"{tag}_status": r.get("status"),
            f"{tag}_validate": v.get("status"),
            f"{tag}_validate_first_nonfinite_step": (v.get("validation") or {}).get("first_nonfinite_step"),
            f"{tag}_load_reserved_MiB": mib((r.get("after_load") or {}).get("reserved_bytes.all.current")),
            f"{tag}_load_peak_allocated_MiB": mib((r.get("after_load") or {}).get("max_allocated")),
            f"{tag}_m1_start_reserved_MiB": mib(s.get("first_measure_start_reserved")),
            f"{tag}_m1_start_allocated_MiB": mib(s.get("first_measure_start_allocated")),
            f"{tag}_m1_peak_reserved_MiB": mib(s.get("first_measure_peak_reserved")),
            f"{tag}_m1_peak_allocated_MiB": mib(s.get("first_measure_peak_allocated")),
            f"{tag}_max_peak_reserved_MiB": mib(s.get("max_peak_reserved")),
            f"{tag}_max_nvml_MiB": mib(s.get("max_nvml_process_peak")),
            f"{tag}_m1_elapsed_s": round(ms[0].get("elapsed_s"), 3) if ms[0].get("elapsed_s") else None,
            f"{tag}_new_tokens": ms[0].get("new_tokens"),
            f"{tag}_output_sha256_m1": ms[0].get("output_sha256"),
            f"{tag}_input_sha256": json.dumps({k: x["sha256"][:16] for k, x in (r.get("inputs") or {}).items()}),
        })
    for k in ("m1_peak_reserved_MiB", "m1_peak_allocated_MiB", "load_reserved_MiB", "max_nvml_MiB"):
        a, b = row.get(f"eager_{k}"), row.get(f"default_{k}")
        row[f"diff_default_minus_eager_{k}"] = None if a is None or b is None else round(b - a, 1)
    row["same_input"] = row["eager_input_sha256"] == row["default_input_sha256"]
    row["same_output_tokens"] = row["eager_output_sha256_m1"] == row["default_output_sha256_m1"]
    rows.append(row)

out = root / "results/pilot_rtx3090_eager_vs_default.csv"
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0]))
    w.writeheader()
    w.writerows(rows)


def f(x):
    return "" if x is None else str(x)


print("### T1")
print("| 조건 | attention (eager/기본) | 1회차 피크 reserved eager | 기본 | 차이 | 1회차 피크 allocated eager | 기본 | 차이 | 로드 후 reserved eager/기본 | NVML 최대 eager/기본 | 검증 eager/기본 | 출력 토큰 동일 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['condition_id']} | {f(r['eager_attn_impl'])}/{f(r['default_attn_impl'])} | {f(r['eager_m1_peak_reserved_MiB'])} | "
          f"{f(r['default_m1_peak_reserved_MiB'])} | {f(r['diff_default_minus_eager_m1_peak_reserved_MiB'])} | "
          f"{f(r['eager_m1_peak_allocated_MiB'])} | {f(r['default_m1_peak_allocated_MiB'])} | "
          f"{f(r['diff_default_minus_eager_m1_peak_allocated_MiB'])} | {f(r['eager_load_reserved_MiB'])}/{f(r['default_load_reserved_MiB'])} | "
          f"{f(r['eager_max_nvml_MiB'])}/{f(r['default_max_nvml_MiB'])} | {f(r['eager_validate'])}/{f(r['default_validate'])} | "
          f"{'예' if r['same_output_tokens'] else '아니오'} |")
print("\n### T1b status")
for r in rows:
    if r["eager_status"] != "ok" or r["default_status"] != "ok" or r["eager_new_tokens"] != r["default_new_tokens"] or not r["same_input"]:
        print("ANOMALY", r["condition_id"], r["eager_status"], r["default_status"], r["eager_new_tokens"], r["default_new_tokens"], r["same_input"])
print(f"\nwritten {out}", file=sys.stderr)
