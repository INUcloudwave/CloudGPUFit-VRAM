#!/usr/bin/env python3
"""Counterfactual replays of one GPU allocator trace into the xMem CachingAllocator (CPU only).

  CUDA_VISIBLE_DEVICES= python xmem_replay_variants.py <xmem_repo> <trace.json> <out.json>

Each variant changes ONE thing in the request stream and reports xMem reserved at measure_0_begin
(= the comparison value, since GPU peak reserved == reserved at measure_0 start in every run so far):
  base                 GPU request stream as recorded (must equal GPU)
  free_on_requested    free at free_requested instead of free_completed (stream-ordering effect)
  no_persistent_ws     drop non-load allocations that are never freed (workspace-like, e.g. cuBLAS/cuBLASLt)
  warmup0_only         stop after warmup_0 (does reserved still grow after the first invocation?)
  no_prefill_logits    drop the full prefill-logits request(s) B*L*V*2 and their frees (what the big
                       prefill temporaries contribute; decoder models only)
"""
import gzip
import json
import logging
import pickle
import sys
from pathlib import Path

xmem_repo, trace_json, out_json = sys.argv[1:4]
sys.path.insert(0, xmem_repo)
logging.disable(logging.CRITICAL)
from perf_estimator.allocator.allocator import CachingAllocator, Device, Stream  # noqa: E402

T = json.loads(Path(trace_json).read_text())
hist = T["history"]
ev = pickle.load(gzip.open(Path(trace_json).with_name(hist["snapshot_file"]), "rb"))["device_traces"][0]
bounds = hist["boundaries"]
cond = T["condition"]
MiB = 2 ** 20

# event index of each boundary (counter matching, as in xmem_replay.py)
pos, na, nf, bi = {}, 0, 0, 0


def mark(i):
    global bi
    while bi < len(bounds) and na == bounds[bi]["allocation.all.allocated"] and nf == bounds[bi]["allocation.all.freed"]:
        pos[bounds[bi]["name"]] = i
        bi += 1


mark(-1)
for i, e in enumerate(ev):
    if e["action"] == "alloc":
        na += 1
    elif e["action"] == "free_completed":
        nf += 1
    mark(i)
load_end, w0_end, m0_begin = pos["model_load_end"], pos["warmup_0_end"], pos["measure_0_begin"]

# persistent non-load allocations (never freed)
freed = set()
alloc_idx = {}
for i, e in enumerate(ev):
    if e["action"] == "alloc":
        alloc_idx[(e["addr"], i)] = e
live_at_end = {}
for i, e in enumerate(ev):
    if e["action"] == "alloc":
        live_at_end[e["addr"]] = i
    elif e["action"] == "free_completed":
        live_at_end.pop(e["addr"], None)
persistent = {i for i in live_at_end.values() if i > load_end}

logits_sz = None
if cond["workload_type"] == "decoder_llm":
    # B*L*V*2 with V from the recorded generation config is not stored; infer: largest request in warmup_0 prefill
    w0 = [e["size"] for e in ev[pos["warmup_0_begin"] + 1: w0_end + 1] if e["action"] == "alloc"]
    logits_sz = max(w0) if w0 else None


def run(variant):
    sim = CachingAllocator(allowed_memory_maximum=T["environment"]["cuda_total_memory"])
    dev, streams, live = Device(index=0), {}, {}
    skip_addr = set()
    res_at = {}
    stop = w0_end if variant == "warmup0_only" else len(ev) - 1
    free_act = "free_requested" if variant == "free_on_requested" else "free_completed"
    for i, e in enumerate(ev[: stop + 1]):
        a = e["action"]
        if a == "alloc":
            if (variant == "no_persistent_ws" and i in persistent) or \
               (variant == "no_prefill_logits" and logits_sz and e["size"] == logits_sz and i > load_end):
                skip_addr.add(e["addr"])
                continue
            s = int(e.get("stream", 0))
            live[e["addr"]] = sim.malloc(dev, streams.setdefault(s, Stream(index=s)), int(e["size"]))
        elif a == free_act:
            if e["addr"] in skip_addr:
                skip_addr.discard(e["addr"])
                continue
            b = live.pop(e["addr"], None)
            if b is not None:
                sim.free(b)
        if i in (load_end, w0_end, m0_begin):
            res_at[i] = sim._trace.max_segment_changes[-1]
    S = sim._trace.max_segment_changes
    return {"after_load_MiB": round(res_at.get(load_end, 0) / MiB, 1),
            "after_warmup0_MiB": round(res_at.get(w0_end, 0) / MiB, 1),
            "at_measure0_begin_MiB": round(res_at[m0_begin] / MiB, 1) if m0_begin in res_at else None,
            "max_reserved_MiB": round(max(S) / MiB, 1)}


gpu = {b["name"]: round(b["reserved_bytes.all.current"] / MiB, 1) for b in bounds}
out = {"condition_id": cond["condition_id"],
       "gpu_reserved_MiB": {k: gpu[k] for k in ("model_load_end", "warmup_0_end", "measure_0_begin")},
       "gpu_measure0_peak_reserved_MiB": round(T["measures"][0]["peak"]["reserved"] / MiB, 1) if T.get("measures") else None,
       "persistent_nonload_allocs": [{"bytes": ev[i]["size"]} for i in sorted(persistent)],
       "prefill_largest_request_bytes": logits_sz,
       "variants": {v: run(v) for v in ("base", "free_on_requested", "no_persistent_ws", "warmup0_only", "no_prefill_logits")}}
Path(out_json).write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
