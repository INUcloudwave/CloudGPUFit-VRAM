#!/usr/bin/env python3
"""Replay a GPU allocator history (trace mode output) into the xMem CachingAllocator (CPU only).

  CUDA_VISIBLE_DEVICES= python xmem_replay.py <xmem_repo> <trace.json> <out.json>

GPU alloc(size = original request) -> xMem malloc(device, stream, size); GPU free_completed -> xMem free(block).
GPU segment_alloc/segment_free are not replayed (xMem decides its own segments) but are counted for comparison.
Boundaries (counter values recorded by the worker after synchronize) are mapped to event positions, and
start/peak/current reserved & allocated of xMem are compared with GPU values at the same positions.
"""
import gzip
import json
import logging
import pickle
import sys
import time
from pathlib import Path

xmem_repo, trace_json, out_json = sys.argv[1:4]
sys.path.insert(0, xmem_repo)
logging.disable(logging.CRITICAL)
from perf_estimator.allocator.allocator import CachingAllocator, Device, Stream  # noqa: E402

T = json.loads(Path(trace_json).read_text())
hist = T["history"]
snap = pickle.load(gzip.open(Path(trace_json).with_name(hist["snapshot_file"]), "rb"))
ev = snap["device_traces"][0]
bounds = hist["boundaries"]

sim = CachingAllocator(allowed_memory_maximum=T["environment"]["cuda_total_memory"])
dev, streams, live = Device(index=0), {}, {}
n_alloc = n_free = n_gpu_seg_alloc = n_gpu_seg_free = missing = 0
gpu_res = gpu_res_peak_window = 0
marks = []          # (n_alloc, n_free) -> xmem trace positions
bi = 0
seg_pos, use_pos = [], []


def at_boundary():
    global bi
    while bi < len(bounds) and n_alloc == bounds[bi]["allocation.all.allocated"] and \
            n_free == bounds[bi]["allocation.all.freed"]:
        marks.append({"name": bounds[bi]["name"], "event_index": idx, "seg_len": len(sim._trace.max_segment_changes),
                      "use_len": len(sim._trace.max_usage_changes), "gpu_reserved_now_from_trace": gpu_res})
        bi += 1


t0 = time.time()
idx = -1
at_boundary()
for idx, e in enumerate(ev):
    a = e["action"]
    if a == "alloc":
        s = int(e.get("stream", 0))
        st = streams.setdefault(s, Stream(index=s))
        live[e["addr"]] = sim.malloc(dev, st, int(e["size"]))
        n_alloc += 1
    elif a == "free_completed":
        b = live.pop(e["addr"], None)
        if b is None:
            missing += 1
        else:
            sim.free(b)
        n_free += 1
    elif a == "segment_alloc":
        n_gpu_seg_alloc += 1
        gpu_res += int(e["size"])
    elif a == "segment_free":
        n_gpu_seg_free += 1
        gpu_res -= int(e["size"])
    at_boundary()

S, U = sim._trace.max_segment_changes, sim._trace.max_usage_changes
bmap = {m["name"]: m for m in marks}
rows = []
gpu_meas = {f"measure_{m['i']}": m for m in T.get("measures", [])}
for name in [b["name"] for b in bounds if b["name"].endswith("_begin")]:
    stem = name[: -len("_begin")]
    b0, b1 = bmap.get(name), bmap.get(stem + "_end")
    if not b0 or not b1:
        rows.append({"window": stem, "error": "boundary not located"})
        continue
    gb0 = next(x for x in bounds if x["name"] == name)
    gb1 = next(x for x in bounds if x["name"] == stem + "_end")
    row = {"window": stem,
           "xmem_start_reserved": S[b0["seg_len"] - 1], "xmem_peak_reserved": max(S[b0["seg_len"] - 1: b1["seg_len"]]),
           "xmem_end_reserved": S[b1["seg_len"] - 1],
           "xmem_start_allocated": U[b0["use_len"] - 1], "xmem_peak_allocated": max(U[b0["use_len"] - 1: b1["use_len"]]),
           "gpu_start_reserved": gb0["reserved_bytes.all.current"], "gpu_end_reserved": gb1["reserved_bytes.all.current"],
           "gpu_start_allocated": gb0["allocated_bytes.all.current"]}
    if stem in gpu_meas:
        row["gpu_peak_reserved"] = gpu_meas[stem]["peak"]["reserved"]
        row["gpu_peak_allocated"] = gpu_meas[stem]["peak"]["allocated"]
    row["note"] = ("gpu_peak_* only for measure windows (peak reset right after *_begin boundary); "
                   "warmup windows compare start/end only")
    rows.append(row)

res = {"condition_id": T["condition"]["condition_id"], "xmem_repo": xmem_repo, "events": len(ev),
       "gpu_alloc_events": n_alloc, "gpu_free_completed_events": n_free, "missing_frees": missing,
       "gpu_segment_alloc_events": n_gpu_seg_alloc, "gpu_segment_free_events": n_gpu_seg_free,
       "xmem_segment_alloc_count": sum(1 for i in range(1, len(S)) if S[i] > S[i - 1]),
       "boundaries_located": len(marks), "boundaries_total": len(bounds),
       "after_load": {"xmem_reserved": S[bmap["model_load_end"]["seg_len"] - 1] if "model_load_end" in bmap else None,
                      "gpu_reserved": next(x for x in bounds if x["name"] == "model_load_end")["reserved_bytes.all.current"]},
       "windows": rows, "replay_wall_s": round(time.time() - t0, 1)}
Path(out_json).write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "windows"}))
for r in rows:
    print(r)
