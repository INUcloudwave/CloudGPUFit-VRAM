#!/usr/bin/env python3
"""Pass/fail check of a pilot (or any list) run on a new host against the RTX 3090 v2 reference.

  python3 -I check_pilot.py <root> <gpu_label> [list.json=config/pilot.json] [--conditions conditions_main395.json]

PASS requires, per condition: measure status ok (5 measures ok, new_tokens == max_new_tokens for text),
validate status ok with all_finite, attn_requested == default, worker sha == v2-verified worker,
input sha256 prefixes == RTX 3090 v2 pilot (reference/pilot_rtx3090_eager_vs_default.csv) when present.
VRAM values are NOT compared (different GPU); they are printed side by side for information only.
"""
import csv
import json
import sys
from pathlib import Path

WORKER_SHA = "d06f0f15982c811c4bc6a24028f71728e461c6ee3651bcfd520ba49b0d675883"
args = [x for x in sys.argv[1:] if not x.startswith("--")]
root, label = Path(args[0]), args[1]
lst = Path(args[2]) if len(args) > 2 else root / "config/pilot.json"
cf = sys.argv[sys.argv.index("--conditions") + 1] if "--conditions" in sys.argv else "conditions_main395.json"
conds = {c["condition_id"]: c for c in json.loads((root / "config" / cf).read_text())["conditions"]}
ids = json.loads(lst.read_text())["condition_ids"]
ref = {}
rp = root / "reference/pilot_rtx3090_eager_vs_default.csv"
if rp.exists():
    ref = {r["condition_id"]: r for r in csv.DictReader(rp.open())}
M = 1 << 20
fails, rows = [], []
for cid in ids:
    d = root / "runs" / label / cid
    why = []
    try:
        r = json.loads((d / "measure.json").read_text())
    except FileNotFoundError:
        r = {}
    try:
        v = json.loads((d / "validate.json").read_text())
    except FileNotFoundError:
        v = {}
    if r.get("status") != "ok":
        why.append(f"measure={r.get('status', 'missing')}")
    if v.get("status") != "ok" or not (v.get("validation") or {}).get("all_finite"):
        why.append(f"validate={v.get('status', 'missing')}")
    if r:
        if (r.get("model") or {}).get("attn_requested") != "default":
            why.append("attn_requested!=default")
        if r.get("worker_sha256") != WORKER_SHA:
            why.append("worker_sha")
        ms = r.get("measures") or []
        c = conds[cid]
        if len(ms) != 5 or not all(m.get("ok") for m in ms):
            why.append("measures!=5ok")
        if "max_new_tokens" in c and any(m.get("new_tokens") != c["max_new_tokens"] for m in ms):
            why.append("new_tokens")
        if cid in ref:
            mine = json.dumps({k: x["sha256"][:16] for k, x in (r.get("inputs") or {}).items()})
            if mine != ref[cid]["default_input_sha256"]:
                why.append("input_sha!=rtx3090")
    s = r.get("summary") or {}
    rows.append((cid, "PASS" if not why else "FAIL " + ",".join(why),
                 round(s["first_measure_peak_reserved"] / M, 1) if s else None,
                 ref.get(cid, {}).get("default_m1_peak_reserved_MiB")))
    if why:
        fails.append(cid)
print("condition | result | m1_peak_reserved_MiB(this host) | m1_peak_reserved_MiB(rtx3090 v2, info only)")
for x in rows:
    print(" | ".join(map(str, x)))
print(f"PILOT_CHECK {'PASS' if not fails else 'FAIL'} {len(ids) - len(fails)}/{len(ids)}")
sys.exit(1 if fails else 0)
