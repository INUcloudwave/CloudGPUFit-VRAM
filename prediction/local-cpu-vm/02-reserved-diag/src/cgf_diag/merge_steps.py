"""Combine per-step (one process per step) CPU history traces into one history directory, and
optionally compare it with a single-process history trace of the same case.

Why: a second torch.profiler session in the same process made RSS grow ~0.6 GB per decode step
(medium fp32, 2026-11-10; profiler off -> no growth). Each step is therefore traced in a fresh
process. The CPU request stream does not depend on CPU allocator state; equivalence with the
single-process trace is checked on a case where both exist (fp16_01).
"""
import argparse
import hashlib
import json
from pathlib import Path

from cgf_diag.events import read_jsonl, write_jsonl

STEP_STRIDE = 10 ** 9


def merge(out):
    out = Path(out)
    dirs = sorted(p for p in (out / "steps").iterdir() if (p / "meta.json").exists())
    metas = [json.loads((d / "meta.json").read_text()) for d in dirs]
    if not metas:
        raise SystemExit("no completed steps")
    ref = metas[0]
    checks = {k: all(m[k] == ref[k] for m in metas) for k in
              ("sequence", "steps", "input_sha256", "resident_storage_sizes", "model_config_sha256", "weights")}
    if not all(checks.values()):
        raise SystemExit(f"steps are not from the same configuration: {checks}")
    expected = list(range(len(ref["steps"])))
    present = [m["step_index"] for m in metas]
    rows, windows, seq = [], [], 0
    for d, m in zip(dirs, metas):
        k = m["step_index"]
        for r in read_jsonl(d / "events.jsonl.gz"):
            r["seq"] = seq
            r["alloc_id"] = k * STEP_STRIDE + r["alloc_id"]
            rows.append(r)
            seq += 1
        for w in json.loads((d / "windows.json").read_text()):
            windows.append({**w, "process_step_index": k, "process_max_rss_bytes": m["max_rss_bytes"]})
    write_jsonl(out / "events.jsonl.gz", rows)
    (out / "windows.json").write_text(json.dumps(windows, indent=1))
    meta = {**ref, "step_index": None, "process_policy": "one fresh process per step, merged in step order",
            "steps_present": present, "steps_missing": sorted(set(expected) - set(present)),
            "unfreed_requests_at_end": sum(m["unfreed_requests_at_end"] for m in metas),
            "unfreed_requests_by_step": {m["step_index"]: m["unfreed_requests_at_end"] for m in metas},
            "max_rss_bytes": max(m["max_rss_bytes"] for m in metas), "consistency_checks": checks,
            "events_file": "events.jsonl.gz"}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta, rows


def step_signature(rows):
    """Per (phase, invocation): request sizes in order + alloc/free pattern using step-local alloc order."""
    sig = {}
    local = {}
    for r in rows:
        k = (r["phase"], r["invocation"])
        s = sig.setdefault(k, {"sizes": [], "pattern": []})
        if r["action"] == "alloc":
            local[r["alloc_id"]] = (k, len(s["sizes"]))
            s["sizes"].append(r["size"])
            s["pattern"].append(("a", len(s["sizes"]) - 1))
        elif r["action"] in ("free", "free_completed"):
            src = local.get(r["alloc_id"])
            s["pattern"].append(("f", src[1] if src and src[0] == k else None))
    return sig


def digest(x):
    return hashlib.sha256(json.dumps(x).encode()).hexdigest()[:16]


def compare(rows_a, rows_b):
    a, b = step_signature(rows_a), step_signature(rows_b)
    out = []
    for k in sorted(set(a) | set(b), key=lambda k: (str(k[0]), k[1] if k[1] is not None else -1)):
        x, y = a.get(k), b.get(k)
        out.append({"phase": k[0], "invocation": k[1],
                    "requests": (len(x["sizes"]) if x else None, len(y["sizes"]) if y else None),
                    "sizes_identical": bool(x and y and x["sizes"] == y["sizes"]),
                    "alloc_free_order_identical": bool(x and y and x["pattern"] == y["pattern"]),
                    "digest": (digest(x["sizes"]) if x else None, digest(y["sizes"]) if y else None)})
    return {"all_identical": all(o["sizes_identical"] and o["alloc_free_order_identical"] for o in out), "steps": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="history dir containing steps/<KK>/")
    ap.add_argument("--compare-with", default=None, help="single-process history dir of the same case")
    a = ap.parse_args()
    meta, rows = merge(a.out)
    res = {"merged_steps": meta["steps_present"], "missing": meta["steps_missing"],
           "unfreed_requests_by_step": meta["unfreed_requests_by_step"], "max_rss_bytes": meta["max_rss_bytes"]}
    if a.compare_with:
        other = list(read_jsonl(Path(a.compare_with) / "events.jsonl.gz"))
        res["equivalence_vs_single_process"] = compare(rows, other)
        (Path(a.out) / "equivalence_vs_single_process.json").write_text(json.dumps(res["equivalence_vs_single_process"], indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "equivalence_vs_single_process"}))
    if a.compare_with:
        print(json.dumps({"all_identical": res["equivalence_vs_single_process"]["all_identical"]}))


if __name__ == "__main__":
    main()
