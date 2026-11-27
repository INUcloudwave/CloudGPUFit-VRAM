"""CPU-side analyses (no GPU data needed).

A. Production baseline trace (original predictor, unchanged, with --trace_output):
   A1 production-equivalent time-grouped xMem replay   -> must equal the stored prediction bytes
   A2 order-preserving sequential replay (Ev Idx order) -> effect of timestamp regrouping
   A3 A2 with the diagnostic PyTorch-2.6 10MiB boundary -> effect of that policy difference
B. GPU-equivalent history trace (cpu_history_trace.py): replays of prefixes of the history into one
   persistent allocator (synthetic resident weights, production placement order), per-step peaks.
All capacities are explicit (default 80 GiB = production gpu_memory_gb, i.e. no capacity pressure).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from cgf_diag.events import cpu_memory_events_from_chrome, normalize_cpu, order_report, read_jsonl
from cgf_diag.replay import Replayer, time_grouped_xmem, timestamp_collision_report, verify_xmem, MIB, GIB

PROD_REGION = "cloudgpufit_inference"


def mib(x):
    return None if x is None else round(x / MIB, 3)


def replay(rows, resident, capacity, policy="xmem_pinned", keep=False):
    r = Replayer(capacity, policy, keep_decisions=keep)
    r.place_resident(resident)
    s = r.run(rows)
    return r, s


def boundary_requests(rows):
    lo, hi = 10 * MIB - 511, 10 * MIB
    c = Counter(r["size"] for r in rows if r["action"] == "alloc" and lo <= r["size"] <= hi)
    return {"count": sum(c.values()), "sizes": dict(c)}


def analyze_baseline(baseline_dir, resident_weights, capacity_gb):
    from inference_trace_runtime import iter_trace_events
    bdir = Path(baseline_dir)
    trace = next(p for p in [bdir / "trace.json.gz", bdir / "trace.json"] if p.exists())
    if trace.suffix == ".gz":
        import gzip, shutil, tempfile
        tmp = Path(tempfile.mkdtemp()) / "trace.json"
        with gzip.open(trace, "rb") as s, tmp.open("wb") as d:
            shutil.copyfileobj(s, d, 1 << 22)
        trace = tmp
    pred = json.loads((bdir / "prediction.json").read_text())
    regions, mem = cpu_memory_events_from_chrome(trace, PROD_REGION, iter_trace_events=iter_trace_events)
    rows, stats, _ = normalize_cpu(mem, regions, phase_of=lambda n: ("baseline/measure", 0))
    b, s = pred["input"]["batch_size"], pred["input"]["seq_len"]
    resident = list(resident_weights) + [b * s * 8, b * s * 8]  # input_ids, attention_mask (int64), production order
    assert sum(resident) - sum(resident_weights) == pred["trace_summary"]["input_storage_bytes"]
    a1 = time_grouped_xmem(rows, resident, capacity_gb, first_region_ts=regions[0][1])
    _, a2 = replay(rows, resident, int(capacity_gb * GIB))
    _, a3 = replay(rows, resident, int(capacity_gb * GIB), "diag_pt26_alloc_size_boundary")
    out = {
        "stored_prediction": {"allocated": pred["predicted_allocated_bytes"], "reserved": pred["predicted_reserved_bytes"]},
        "A1_time_grouped_production_equivalent": {"allocated": a1[0], "reserved": a1[1],
                                                  "matches_stored": (a1[0], a1[1]) == (pred["predicted_allocated_bytes"], pred["predicted_reserved_bytes"])},
        "A2_sequential_xmem": {"allocated": a2["peak_allocated_bytes"], "reserved": a2["peak_reserved_bytes"]},
        "A3_sequential_diag_pt26_boundary": {"allocated": a3["peak_allocated_bytes"], "reserved": a3["peak_reserved_bytes"]},
        "requests": sum(1 for r in rows if r["action"] == "alloc"), "pairing": stats, "order": order_report(mem),
        "timestamp_groups": timestamp_collision_report(rows), "requests_rounding_to_exactly_10MiB": boundary_requests(rows),
        "tensor_lifetime_count_stored": pred["trace_summary"]["tensor_lifetime_count"],
    }
    return out, rows


def size_seq(rows, phase, inv):
    return [r["size"] for r in rows if r["action"] == "alloc" and r["phase"] == phase and r["invocation"] == inv]


def analyze_history(history_dir, capacity_gb, baseline_rows=None):
    hdir = Path(history_dir)
    meta = json.loads((hdir / "meta.json").read_text())
    windows = json.loads((hdir / "windows.json").read_text())
    rows = list(read_jsonl(hdir / meta["events_file"]))
    resident = meta["resident_storage_sizes"]
    cap = int(capacity_gb * GIB)
    # end seq of each step
    step_keys = []
    for r in rows:
        k = (r["phase"], r["invocation"])
        if not step_keys or step_keys[-1] != k:
            step_keys.append(k)
    out = {"meta": {k: meta[k] for k in ("case_id", "sequence", "weights", "steps", "input_sha256", "resident_storage_bytes",
                                         "unfreed_requests_at_end", "environment", "gc_policy")},
           "windows": [{k: w.get(k) for k in ("region", "seconds", "requests", "frees", "request_bytes", "live_requests_after_step",
                                              "actual_new_tokens_per_sample", "finite", "gc_collected_objects", "max_rss_bytes")}
                       | {"order": w["order"], "pairing": w["pairing"]} for w in windows]}
    # repetition identity of generate request sequences
    gens = [k for k in step_keys if k[0].endswith(("warmup", "measure"))]
    ref = size_seq(rows, *gens[0]) if gens else []
    out["generate_request_sequence_identity"] = [
        {"step": f"{k[0]}/{k[1]}", "requests": len(s), "identical_to_first_generate": s == ref,
         "multiset_identical": Counter(s) == Counter(ref)} for k in gens for s in [size_seq(rows, *k)]]
    if baseline_rows is not None and gens:
        bseq = [r["size"] for r in baseline_rows if r["action"] == "alloc"]
        g = size_seq(rows, *gens[0])
        cb, cg = Counter(bseq), Counter(g)
        out["first_generate_vs_production_baseline_requests"] = {
            "baseline_requests": len(bseq), "history_first_generate_requests": len(g), "sequence_identical": bseq == g,
            "only_in_history": dict((cg - cb).most_common(10)), "only_in_baseline": dict((cb - cg).most_common(10))}
    # prefix replays
    replays = {}
    for policy in ("xmem_pinned", "diag_pt26_alloc_size_boundary"):
        r, s = replay(rows, resident, cap, policy)
        replays[policy] = s
    out["full_history_replay"] = {p: {"peak_reserved": s["peak_reserved_bytes"], "peak_allocated": s["peak_allocated_bytes"],
                                      "final_reserved": s["final_reserved_bytes"], "oom": s["oom"],
                                      "unknown_frees": s["unknown_frees"], "live_blocks_at_end": s["live_blocks_at_end"],
                                      "per_step": [{"step": f"{ph['phase']}/{ph['invocation']}",
                                                    "reserved_at_start": ph["reserved_at_start"],
                                                    "peak_reserved": ph["peak_reserved"], "peak_allocated": ph["peak_allocated"],
                                                    "reserved_at_end": ph.get("reserved_at_end"),
                                                    "new_segment_bytes": ph["new_segment_bytes"], "requests": ph["requests"]}
                                                   for ph in s["phases"]]}
                                  for p, s in replays.items()}
    # isolated single generate (first generate window only, resident weights only)
    if gens:
        only = [r for r in rows if (r["phase"], r["invocation"]) == gens[0]]
        _, s1 = replay(only, resident, cap)
        out["isolated_first_generate"] = {"peak_reserved": s1["peak_reserved_bytes"], "peak_allocated": s1["peak_allocated_bytes"],
                                          "unknown_frees": s1["unknown_frees"]}
    # GPU-comparable measured-window peaks (GPU resets peaks at measurement start; reserved peak >= current)
    for p, s in replays.items():
        agg = {}
        for ph in s["phases"]:
            if ph["phase"] and ph["phase"].endswith("measure"):
                mode = ph["phase"].split("/")[0]
                cur = agg.setdefault(mode, {"peak_reserved": 0, "peak_allocated": 0, "reserved_at_first_measure_start": None})
                if cur["reserved_at_first_measure_start"] is None:
                    cur["reserved_at_first_measure_start"] = ph["reserved_at_start"]
                cur["peak_reserved"] = max(cur["peak_reserved"], ph["peak_reserved"], ph["reserved_at_start"])
                cur["peak_allocated"] = max(cur["peak_allocated"], ph["peak_allocated"])
        out["full_history_replay"][p]["measured_window_peaks_by_mode"] = agg
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--baseline-dir")
    ap.add_argument("--history-dir", action="append", default=[])
    ap.add_argument("--capacity-gb", type=float, default=80.0)
    ap.add_argument("--xmem-root", default="/work/xMem")
    ap.add_argument("--xmem-lock", default="/exp/provenance/xmem-allocator-lock.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"case_id": a.case, "capacity_gb": a.capacity_gb, "capacity_unit": "GiB (x 2^30), xMem allowed_memory_maximum",
           "xmem_verification": verify_xmem(a.xmem_root, a.xmem_lock)}
    baseline_rows = None
    resident_weights = None
    metas = [json.loads((Path(h) / "meta.json").read_text()) for h in a.history_dir]
    if metas:
        resident_weights = metas[0]["resident_storage_sizes"]
    if a.baseline_dir:
        if resident_weights is None:
            raise SystemExit("resident weight sizes come from a history meta.json (same from_config build)")
        res["baseline"], baseline_rows = analyze_baseline(a.baseline_dir, resident_weights, a.capacity_gb)
    res["history"] = {Path(h).name + ":" + m["sequence"]: analyze_history(h, a.capacity_gb, baseline_rows) for h, m in zip(a.history_dir, metas)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"written": a.out}))


if __name__ == "__main__":
    main()
