#!/usr/bin/env python3
"""Compare two per-condition CSVs (export_for_inuos.py format) and summarise "outside" (NVML peak - peak reserved). CPU only.

  python3 -I tools/compare_gpu_runs.py <base.csv> <other.csv> <out_prefix> [label_base] [label_other]

Writes <out_prefix>_per_condition.csv (status pair, m1 peak reserved/allocated of both and the difference in bytes,
outside of both), <out_prefix>_summary.md (status table, identical counts and difference ranges per model,
outside distribution of both). Used 2026-11-18 for RTX 3090 cu124 vs cu128 (version effect) and outside per GPU.
"""
import collections
import csv
import sys

base, other, outp = sys.argv[1:4]
lb, lo = (sys.argv[4:6] + ["base", "other"])[:2] if len(sys.argv) > 4 else ("base", "other")
A = {r["condition_id"]: r for r in csv.DictReader(open(base))}
B = {r["condition_id"]: r for r in csv.DictReader(open(other))}
M = 2 ** 20
I = lambda v: int(v) if v not in (None, "") else None  # noqa: E731
rows, st, by, outs = [], collections.Counter(), collections.defaultdict(lambda: [0, 0, [], []]), {lb: collections.Counter(), lo: collections.Counter()}
for cid in A:
    a, b = A[cid], B.get(cid, {})
    st[(a["measure_status"], b.get("measure_status"))] += 1
    x = {"condition_id": cid, f"{lb}_status": a["measure_status"], f"{lo}_status": b.get("measure_status")}
    for k in ("m1_peak_reserved", "m1_peak_allocated", "max_nvml_process_peak"):
        x[f"{lb}_{k}"], x[f"{lo}_{k}"] = I(a.get(k)), I(b.get(k))
        x[f"d_{k}"] = (x[f"{lo}_{k}"] - x[f"{lb}_{k}"]) if None not in (x[f"{lb}_{k}"], x[f"{lo}_{k}"]) else None
    for lab, r in ((lb, a), (lo, b)):
        n, p = I(r.get("max_nvml_process_peak")), I(r.get("m1_peak_reserved"))
        x[f"{lab}_outside"] = (n - p) if None not in (n, p) else None
        if x[f"{lab}_outside"] is not None:
            outs[lab][round(x[f"{lab}_outside"] / M)] += 1
    if x["d_m1_peak_reserved"] is not None:
        m = by[cid.split("__")[0]]
        m[0] += 1
        m[1] += x["d_m1_peak_reserved"] == 0
        if x["d_m1_peak_reserved"]:
            m[2].append(x["d_m1_peak_reserved"] / M)
        if x["d_m1_peak_allocated"]:
            m[3].append(x["d_m1_peak_allocated"] / M)
    rows.append(x)
with open(f"{outp}_per_condition.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
md = [f"# {lb} vs {lo}", "", "## 상태 (base, other) -> 조건 수", ""] + [f"- {k}: {v}" for k, v in st.items()]
tot, same = sum(v[0] for v in by.values()), sum(v[1] for v in by.values())
md += ["", f"## peak reserved 바이트 단위 동일: {same}/{tot}", "", "| 모델 | 동일 | reserved 차이 MiB (정렬) | allocated가 다른 조건 수, 범위 MiB |", "|---|---|---|---|"]
for m, (n, s, d, da) in sorted(by.items()):
    md.append(f"| {m} | {s}/{n} | {', '.join(f'{v:+.0f}' for v in sorted(d)) or '-'} | {len(da)}" + (f" ({min(da):+.1f} ~ {max(da):+.1f})" if da else "") + " |")
md += ["", "## outside = max NVML process peak - m1 peak reserved (MiB -> 조건 수)", ""]
for lab in (lb, lo):
    md.append(f"- {lab}: {dict(sorted(outs[lab].items()))}")
open(f"{outp}_summary.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
