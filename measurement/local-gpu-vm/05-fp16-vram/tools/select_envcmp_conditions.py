#!/usr/bin/env python3
"""Pick light / medium / heavy conditions per model from an existing L256 run (g4dn.xlarge T4) for the env comparison.

  python3 -I select_envcmp_conditions.py <runs/<label> of the reference L256 run> <config dir> <out_prefix>
         [--heavy-max-frac 0.85] [--wall-cap-s 600]

Inputs per condition: <cid>/measure.json (status, summary, measures[].stats_end.num_alloc_retries, environment),
<cid>/measure.launch.json (wall_s = whole fresh process incl. model load), <cid>/validate.json if present.
Rules (deterministic, applied to measure status == ok only):
  light  : batch_size == 1 if the model has an ok batch-1 condition (else the smallest batch); among those the smallest
           first-measure peak reserved; tie -> smaller input (resolution or L+N), then shorter wall.
  heavy  : largest first-measure peak reserved with (a) no allocator retry in any measure iteration (retry-zone values are
           path dependent), (b) peak reserved <= heavy-max-frac x cuda_total_memory, (c) wall_s <= wall-cap-s;
           tie -> shorter wall.
  medium : the remaining ok condition (same retry/wall filters) whose peak reserved is closest to the geometric mean of
           light and heavy (log scale); tie -> shorter wall. Must differ from light and heavy.
  ViT keeps its fixed resolution (224) automatically: only batch differs in the manifest.
Writes <out_prefix>.json (condition_ids, 39 expected), <out_prefix>.csv/.md (table with reasons), and
config-ready manifest <out_prefix>_conditions.json (the selected condition objects copied byte-identically from the
reference manifest, so inputs/seeds are unchanged).
"""
import csv
import json
import math
import sys
from pathlib import Path

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opt = {sys.argv[i][2:]: sys.argv[i + 1] for i in range(1, len(sys.argv) - 1) if sys.argv[i].startswith("--")}
runs, cfgdir, outp = Path(args[0]), Path(args[1]), Path(args[2])
heavy_frac = float(opt.get("heavy-max-frac", 0.85))
wall_cap = float(opt.get("wall-cap-s", 600))
manifest = json.loads((cfgdir / "conditions_main395_l256.json").read_text())
conds = {c["condition_id"]: c for c in manifest["conditions"]}
MiB = 2 ** 20

rows = {}
for cid, c in conds.items():
    d = runs / cid
    try:
        r = json.loads((d / "measure.json").read_text())
    except FileNotFoundError:
        continue
    la = json.loads((d / "measure.launch.json").read_text()) if (d / "measure.launch.json").exists() else {}
    v = json.loads((d / "validate.json").read_text()) if (d / "validate.json").exists() else {}
    s = r.get("summary") or {}
    retries = max([((m.get("stats_end") or {}).get("num_alloc_retries") or 0) for m in r.get("measures") or []] or [0])
    rows[cid] = {"cid": cid, "model": c["model"], "kind": c["workload_type"], "B": c["batch_size"],
                 "L": c.get("input_len"), "N": c.get("max_new_tokens"), "R": c.get("resolution"),
                 "status": r.get("status"), "validate": v.get("status"),
                 "peak_res": s.get("first_measure_peak_reserved"), "peak_alloc": s.get("first_measure_peak_allocated"),
                 "total": (r.get("environment") or {}).get("cuda_total_memory"),
                 "gpu": (r.get("environment") or {}).get("gpu_name"), "driver": (r.get("environment") or {}).get("driver"),
                 "retries": retries, "wall": la.get("wall_s"), "worker_wall": r.get("worker_wall_s")}

by_model = {}
for x in rows.values():
    by_model.setdefault(x["model"], []).append(x)
models = sorted({c["model"] for c in conds.values()}, key=lambda m: [c["model"] for c in manifest["conditions"]].index(m))
sel, table, problems = [], [], []
for m in models:
    ok = [x for x in by_model.get(m, []) if x["status"] == "ok" and x["peak_res"] is not None]
    if not ok:
        problems.append(f"{m}: no ok condition in reference run")
        continue
    w = lambda x: x["wall"] if x["wall"] is not None else 1e9  # noqa: E731
    size = lambda x: x["R"] if x["R"] else x["L"] + x["N"]  # noqa: E731  (input size tie-break)
    b1 = [x for x in ok if x["B"] == 1] or [x for x in ok if x["B"] == min(y["B"] for y in ok)]
    light = min(b1, key=lambda x: (x["peak_res"], size(x), w(x)))
    stable = [x for x in ok if x["retries"] == 0 and w(x) <= wall_cap and x["total"]
              and x["peak_res"] <= heavy_frac * x["total"] and x["cid"] != light["cid"]]
    if not stable:
        problems.append(f"{m}: no heavy candidate under filters")
        continue
    heavy = max(stable, key=lambda x: (x["peak_res"], -size(x), -w(x)))
    target = math.sqrt(light["peak_res"] * heavy["peak_res"])
    mids = [x for x in stable if x["cid"] != heavy["cid"]]
    if not mids:
        problems.append(f"{m}: no medium candidate")
        continue
    mid = min(mids, key=lambda x: (abs(math.log(x["peak_res"] / target)), size(x), w(x)))
    for tier, x, why in (
            ("light", light, f"batch {light['B']}, smallest peak reserved among {'batch-1' if light['B'] == 1 else 'smallest-batch'} ok conditions"),
            ("medium", mid, f"peak reserved closest to geometric mean of light/heavy ({target / MiB:.0f} MiB); no retry; wall <= {wall_cap:.0f}s"),
            ("heavy", heavy, f"largest peak reserved with no allocator retry, <= {heavy_frac:.0%} of {heavy['total'] / MiB:.0f} MiB, wall <= {wall_cap:.0f}s")):
        sel.append(x["cid"])
        table.append({"model": m, "tier": tier, "condition_id": x["cid"], "batch": x["B"],
                      "input_len": x["L"], "new_tokens": x["N"], "resolution": x["R"],
                      "ref_peak_reserved_MiB": round(x["peak_res"] / MiB, 1),
                      "ref_peak_allocated_MiB": round(x["peak_alloc"] / MiB, 1),
                      "ref_wall_s": x["wall"], "ref_retries": x["retries"], "ref_validate": x["validate"], "reason": why})

outp.parent.mkdir(parents=True, exist_ok=True)
ref = next(iter(rows.values()), {})
meta = {"reference_run": str(runs), "reference_gpu": ref.get("gpu"), "reference_driver": ref.get("driver"),
        "rules": {"heavy_max_frac": heavy_frac, "wall_cap_s": wall_cap}, "n_models": len(models),
        "n_selected": len(sel), "n_unique": len(set(sel)), "problems": problems,
        "ref_total_wall_s_selected": round(sum(t["ref_wall_s"] or 0 for t in table), 1)}
(outp.with_suffix(".json")).write_text(json.dumps({**meta, "condition_ids": sel}, indent=1))
with outp.with_suffix(".csv").open("w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(table[0].keys()) if table else ["model"])
    wr.writeheader()
    wr.writerows(table)
hdr = "| 모델 | 구분 | condition_id | 배치 | 입력/생성 또는 해상도 | 기존 peak reserved (MiB) | 기존 실행 시간 (s) | 선정 이유 |"
lines = [hdr, "|" + "---|" * 8]
for t in table:
    shape = f"{t['input_len']}/{t['new_tokens']}" if t["input_len"] else f"r{t['resolution']}"
    lines.append(f"| {t['model'].split('/')[-1]} | {t['tier']} | `{t['condition_id']}` | {t['batch']} | {shape} | "
                 f"{t['ref_peak_reserved_MiB']:,} | {t['ref_wall_s']} | {t['reason']} |")
outp.with_suffix(".md").write_text("\n".join(lines) + "\n")
sub = {**{k: v for k, v in manifest.items() if k != "conditions"},
       "selection": {"source_manifest": "conditions_main395_l256.json", **meta},
       "conditions": [conds[c] for c in sel]}
Path(str(outp) + "_conditions.json").write_text(json.dumps(sub, indent=1))
print(json.dumps(meta, indent=1))
print("\n".join(lines))
sys.exit(0 if len(sel) == 3 * len(models) == len(set(sel)) and not problems else 1)
