"""Stage 2/3: import a GPU memory-history run, replay its requests into xMem, compare decisions.

Inputs (one fresh-process run directory produced by the inugpu collector, see handoff/):
  memory_snapshot.pickle[.gz]  torch.cuda.memory._snapshot() at end of run (trusted experiment output)
  boundaries.jsonl             collector markers with cumulative counters + memory_stats
  run_meta.json                environment / case / run_id
Outputs (derived; originals untouched):
  gpu_events.normalized.jsonl.gz, import_report.json, decisions_<policy>.csv.gz,
  comparison_<policy>.json (first divergence + context, per-phase table),
  [request_diff_vs_cpu.json, hybrid_<...>.json when --cpu-history is given]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from cgf_diag.events import load_snapshot, normalize_gpu_snapshot, write_jsonl, read_jsonl, gpu_reserved_timeline
from cgf_diag.replay import Replayer, verify_xmem, DECISION_FIELDS, MIB, GIB


def read_boundaries(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return rows


def insert_empty_cache_rows(rows, placed, markers):
    """Turn placed 'empty_cache_before' markers into explicit empty_cache_call request rows."""
    pos = sorted([(p["position"], m) for p, m in zip(placed, markers) if m.get("kind") == "empty_cache_before"],
                 key=lambda x: x[0])
    if not pos:
        return rows, 0

    def row(seq, m):
        return {"seq": seq, "action": "empty_cache_call", "alloc_id": None, "size": 0, "stream": None,
                "phase": m.get("phase"), "invocation": m.get("invocation"), "src": "collector_marker"}

    out, k = [], 0
    for i, r in enumerate(rows):
        while k < len(pos) and pos[k][0] == i:
            out.append(row(r["seq"] - 0.5, pos[k][1]))
            k += 1
        out.append(r)
    while k < len(pos):
        out.append(row((rows[-1]["seq"] if rows else 0) + 0.5, pos[k][1]))
        k += 1
    return out, len(pos)


def observed_decisions(rows):
    """Per GPU alloc: segment bytes created/released since the previous request event (observed)."""
    out = {}
    pend_alloc = pend_free = 0
    in_empty_cache = False  # segment frees right after an explicit empty_cache call belong to that call
    timeline = gpu_reserved_timeline(rows)
    for r, res in zip(rows, timeline):
        a = r["action"]
        if a in ("segment_alloc", "segment_map"):
            in_empty_cache = False
            pend_alloc += r["size"]
        elif a in ("segment_free", "segment_unmap"):
            if not in_empty_cache:
                pend_free += r["size"]
        elif a == "alloc":
            out[r["seq"]] = {"new_segment_bytes": pend_alloc, "released_segment_bytes": pend_free, "reserved_after": res}
            pend_alloc = pend_free = 0
            in_empty_cache = False
        elif a in ("free_requested", "free_completed"):
            pend_alloc = pend_free = 0
            in_empty_cache = False
        elif a == "empty_cache_call":
            pend_alloc = pend_free = 0
            in_empty_cache = True
    return out


def compare(rows, replayer, context=8):
    obs = observed_decisions(rows)
    dec = {d[0]: dict(zip(DECISION_FIELDS, d)) for d in replayer.decisions}
    allocs = [r for r in rows if r["action"] == "alloc"]
    first_decision = first_reserved = None
    mism = 0
    table = []
    for i, r in enumerate(allocs):
        o, d = obs.get(r["seq"]), dec.get(r["seq"])
        if d is None:
            break  # replay stopped (OOM)
        same = o["new_segment_bytes"] == d["new_segment_bytes"] and o["released_segment_bytes"] == d["released_segment_bytes"]
        if not same:
            mism += 1
            if first_decision is None:
                first_decision = i
        if first_reserved is None and o["reserved_after"] != d["reserved_after"]:
            first_reserved = i
        table.append((i, r, o, d))

    def ctx(i):
        if i is None:
            return None
        lo, hi = max(0, i - context), min(len(table), i + context + 1)
        return [{"request_index": j, "seq": t[1]["seq"], "phase": t[1]["phase"], "invocation": t[1]["invocation"],
                 "request_bytes": t[1]["size"], "stream": t[1]["stream"],
                 "gpu_new_segment": t[2]["new_segment_bytes"], "xmem_new_segment": t[3]["new_segment_bytes"],
                 "gpu_released": t[2]["released_segment_bytes"], "xmem_released": t[3]["released_segment_bytes"],
                 "gpu_reserved_after": t[2]["reserved_after"], "xmem_reserved_after": t[3]["reserved_after"],
                 "xmem_block_bytes": t[3]["block_bytes"]} for j, t in enumerate(table[lo:hi], start=lo)]

    return {"requests_compared": len(table), "decision_mismatches": mism,
            "first_decision_divergence_request_index": first_decision, "first_reserved_divergence_request_index": first_reserved,
            "context_first_decision_divergence": ctx(first_decision), "context_first_reserved_divergence": ctx(first_reserved),
            "observed_peak_reserved_trace": max(gpu_reserved_timeline(rows) or [0])}


def phase_table(rows, replayer_summary, boundaries):
    """Observed (collector memory_stats at boundaries + trace timeline) vs replay, per phase/invocation."""
    tl = gpu_reserved_timeline(rows)
    obs = defaultdict(lambda: {"peak_reserved_trace": 0})
    for r, res in zip(rows, tl):
        k = (r["phase"], r["invocation"])
        obs[k]["peak_reserved_trace"] = max(obs[k]["peak_reserved_trace"], res)
    stats = {}
    for b in boundaries:
        if b.get("kind") in ("phase_end",):
            stats[(b.get("phase"), b.get("invocation"))] = b.get("memory_stats", {})
    out = []
    for ph in replayer_summary["phases"]:
        k = (ph["phase"], ph["invocation"])
        ms = stats.get(k, {})
        out.append({"phase": k[0], "invocation": k[1],
                    "gpu_peak_reserved_trace": obs[k]["peak_reserved_trace"] if k in obs else None,
                    "gpu_reserved_peak_stat_at_end": ms.get("reserved_bytes.all.peak"),
                    "gpu_allocated_peak_stat_at_end": ms.get("allocated_bytes.all.peak"),
                    "xmem_peak_reserved": ph["peak_reserved"], "xmem_peak_allocated": ph["peak_allocated"],
                    "xmem_reserved_at_start": ph["reserved_at_start"], "xmem_reserved_at_end": ph.get("reserved_at_end")})
    return out


def request_diff(gpu_rows, cpu_rows, frames_by_seq, top=15):
    """CPU vs GPU request streams per (phase, invocation): counts, multiset diffs, first sequence mismatch."""
    def group(rows):
        g = defaultdict(list)
        for r in rows:
            if r["action"] == "alloc":
                g[(r["phase"], r["invocation"])].append(r)
        return g
    G, C = group(gpu_rows), group(cpu_rows)
    out = []
    for k in sorted(set(G) | set(C), key=lambda k: (str(k[0]), k[1] if k[1] is not None else -1)):
        g, c = G.get(k, []), C.get(k, [])
        gs, cs = [r["size"] for r in g], [r["size"] for r in c]
        first = next((i for i, (x, y) in enumerate(zip(gs, cs)) if x != y), None if len(gs) == len(cs) else min(len(gs), len(cs)))
        gm, cm = Counter(gs), Counter(cs)
        only_g = (gm - cm).most_common(top)
        examples = {}
        for size, _ in only_g[:5]:
            ex = next((r for r in g if r["size"] == size), None)
            if ex is not None and ex["seq"] in frames_by_seq:
                examples[str(size)] = frames_by_seq[ex["seq"]]
        out.append({"phase": k[0], "invocation": k[1], "gpu_requests": len(gs), "cpu_requests": len(cs),
                    "gpu_request_bytes": sum(gs), "cpu_request_bytes": sum(cs), "sequence_identical": gs == cs,
                    "first_sequence_mismatch_index": first,
                    "around_first_mismatch": None if first is None else {"gpu": gs[max(0, first - 3):first + 5], "cpu": cs[max(0, first - 3):first + 5]},
                    "only_in_gpu_top": only_g, "only_in_cpu_top": (cm - gm).most_common(top), "gpu_only_examples_frames": examples})
    return out


def hybrid_rows(gpu_rows, cpu_rows, gpu_prefix_phases):
    """GPU-recorded rows for the given phases (e.g. model_load) followed by CPU inference rows.
    Resident memory then comes from the GPU-recorded load: no synthetic residency is added."""
    pre = [r for r in gpu_rows if r["phase"] in gpu_prefix_phases]
    # drop frees (in the GPU prefix) of allocations made outside the prefix -- none expected
    base = (max((r["seq"] for r in pre), default=0) + 1)
    out = [dict(r) for r in pre]
    for r in cpu_rows:
        rr = dict(r)
        rr["seq"] = base + r["seq"]
        rr["alloc_id"] = ("cpu", r["alloc_id"])
        out.append(rr)
    for r in out[:len(pre)]:
        r["alloc_id"] = ("gpu", r["alloc_id"]) if r["alloc_id"] is not None else None
    return out


def select_cpu_rows(rows, select):
    """select: list of 'src_phase:inv=dst_phase:inv' (order kept). Frees whose allocation is not
    selected are dropped (replayer would count them as unknown anyway)."""
    if not select:
        return rows
    mapping = {}
    for item in select:
        src, dst = item.split("=")
        sp, si = src.rsplit(":", 1)
        dp, di = dst.rsplit(":", 1)
        mapping[(sp, int(si))] = (dp, int(di))
    keep_ids, out = set(), []
    for r in rows:
        k = (r["phase"], r["invocation"])
        if r["action"] == "alloc" and k in mapping:
            keep_ids.add(r["alloc_id"])
        if r["alloc_id"] in keep_ids and (k in mapping or r["action"] != "alloc"):
            rr = dict(r)
            rr["phase"], rr["invocation"] = mapping.get(k, (r["phase"], r["invocation"]))
            out.append(rr)
    return out


# GPU run_id -> CPU history windows (aws_replica_local_core trace) with identical call sequence
CPU_SELECT_PRESETS = {
    "R0_aws_replica": [],  # identity: same phase names
    "R1_controlled": ["performance/finite_check:0=controlled/finite_check:0", "performance/warmup:0=controlled/warmup:0",
                      "performance/measure:0=controlled/measure:0", "performance/measure:1=controlled/measure:1",
                      "performance/measure:2=controlled/measure:2"],
    "R2_cold_single": ["performance/warmup:0=single/measure:0"],
    "D1_cold_single_empty_cache": ["performance/warmup:0=single/measure:0"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="one GPU run directory (fresh process)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--capacity-gb", type=float, default=80.0, help="no-pressure policy comparison (GiB)")
    ap.add_argument("--capacity-bytes", type=int, default=None, help="explicit allocator capacity for pressure tests")
    ap.add_argument("--cpu-history", default=None, help="cpu_history_trace output dir for the same case")
    ap.add_argument("--gpu-prefix-phases", default="model_load,empty_cache", help="comma list for hybrid replay")
    ap.add_argument("--cpu-select", action="append", default=None,
                    help="'src_phase:inv=dst_phase:inv' (repeatable). Default: preset for the run_id")
    ap.add_argument("--xmem-root", default="/work/xMem")
    ap.add_argument("--xmem-lock", default="/exp/provenance/xmem-allocator-lock.json")
    a = ap.parse_args()
    run = Path(a.run_dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"xmem_verification": verify_xmem(a.xmem_root, a.xmem_lock)}
    meta = json.loads((run / "run_meta.json").read_text())
    snap_path = next(p for p in [run / "memory_snapshot.pickle", run / "memory_snapshot.pickle.gz"] if p.exists())
    boundaries = read_boundaries(run / "boundaries.jsonl")
    snap = load_snapshot(snap_path)
    markers = [b for b in boundaries if b.get("kind") in ("phase_begin", "empty_cache_before")]
    rows, istats, frames = normalize_gpu_snapshot(snap, boundaries=markers)
    rows, n_ec = insert_empty_cache_rows(rows, istats["boundaries_placed"], markers)
    write_jsonl(out / "gpu_events.normalized.jsonl.gz", rows)
    frames_by_seq = {f["seq"]: f["frames"] for f in frames}
    start = next((b for b in boundaries if b.get("kind") == "record_start"), {})
    end = next((b for b in reversed(boundaries) if b.get("kind") == "record_end"), {})
    n_alloc = istats["actions"].get("alloc", 0)
    end_count = end.get("counters", {}).get("alloc")
    streams = sorted({r["stream"] for r in rows if r["action"] == "alloc"}, key=str)
    report["import"] = {**istats, "empty_cache_markers": n_ec, "streams": [str(s) for s in streams],
                        "initial_state": "empty" if start.get("memory_stats", {}).get("reserved_bytes.all.current") == 0
                        and start.get("counters", {}).get("alloc") == 0 else "NOT_EMPTY_OR_UNKNOWN",
                        "trace_complete": end_count is not None and n_alloc == end_count - start.get("counters", {}).get("alloc", 0),
                        "alloc_events": n_alloc, "alloc_counter_end": end_count}
    if report["import"]["initial_state"] != "empty":
        report["warning"] = "Recording did not start from an empty allocator: results are NOT a full-history comparison."
    if not report["import"]["trace_complete"]:
        report["warning_truncation"] = "alloc events != allocation counter delta: trace truncated or counters inconsistent."
    if len(streams) > 1:
        report["warning_streams"] = "multiple streams: xMem reuses other-stream blocks (see synthetic test) -> decisions after first cross-stream reuse are not policy-equivalent."
    cap = a.capacity_bytes if a.capacity_bytes else int(a.capacity_gb * GIB)
    report["capacity_bytes"] = cap
    report["policies"] = {}
    for policy in ("xmem_pinned", "diag_pt26_alloc_size_boundary"):
        r = Replayer(cap, policy, keep_decisions=True)
        s = r.run(rows)
        with gzip.open(out / f"decisions_{policy}.csv.gz", "wt", newline="") as f:
            w = csv.writer(f)
            w.writerow(DECISION_FIELDS)
            w.writerows(r.decisions)
        cmp = compare(rows, r)
        report["policies"][policy] = {"replay": {k: v for k, v in s.items() if k != "phases"}, "comparison": cmp,
                                      "phase_table": phase_table(rows, s, boundaries)}
    report["observed_final_memory_stats"] = end.get("memory_stats")
    if a.cpu_history:
        cpu_rows = list(read_jsonl(Path(a.cpu_history) / "events.jsonl.gz"))
        select = a.cpu_select if a.cpu_select is not None else CPU_SELECT_PRESETS.get(meta.get("manifest_run_id") or meta.get("run_id"), [])
        cpu_rows = select_cpu_rows(cpu_rows, select)
        report["cpu_selection"] = {"cpu_history": str(a.cpu_history), "select": select, "rows": len(cpu_rows)}
        report["request_diff_vs_cpu"] = request_diff(rows, cpu_rows, frames_by_seq)
        prefix = set(a.gpu_prefix_phases.split(","))
        hy = hybrid_rows(rows, cpu_rows, prefix)
        r = Replayer(cap, "xmem_pinned")
        s = r.run(hy)
        report["hybrid_gpu_prefix_then_cpu_requests"] = {"gpu_prefix_phases": sorted(prefix),
                                                         "peak_reserved": s["peak_reserved_bytes"], "peak_allocated": s["peak_allocated_bytes"],
                                                         "phases": s["phases"], "oom": s["oom"]}
    (out / "import_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({"written": str(out / "import_report.json"), "run_id": meta.get("run_id"), "case_id": meta.get("case_id")}))


if __name__ == "__main__":
    main()
