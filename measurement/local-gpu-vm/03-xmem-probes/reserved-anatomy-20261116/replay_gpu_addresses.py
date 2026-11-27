#!/usr/bin/env python3
"""Hypothesis test: xMem divergence comes from segment ADDRESS order (tie-break among equal-size free blocks).
Replay the GPU request stream, but give every new xMem segment the address of the next GPU segment_alloc.
  CUDA_VISIBLE_DEVICES= python replay_gpu_addresses.py <xmem_repo> <trace.json>"""
import gzip, json, logging, pickle, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1]); logging.disable(logging.CRITICAL)
from perf_estimator.allocator.allocator import CachingAllocator, Device, Stream, Block
T = json.loads(Path(sys.argv[2]).read_text()); MiB = 2**20
ev = pickle.load(gzip.open(Path(sys.argv[2]).with_name(T["history"]["snapshot_file"]), "rb"))["device_traces"][0]
gseg = [(e["addr"], e["size"]) for e in ev if e["action"] == "segment_alloc"]
sim = CachingAllocator(allowed_memory_maximum=T["environment"]["cuda_total_memory"])
q = iter(gseg); mism = []
class Fake:
    def __init__(self, a): self.address = a
def gmalloc(size):
    a, s = next(q, (None, None))
    if a is None: return None
    if s != size: mism.append((size, s))
    return Fake(a)
sim._gpu_device.malloc = gmalloc
sim._gpu_device.addr_free = lambda addr: None
dev, st, live = Device(index=0), {}, {}
gres = 0; first = None; na = 0
for i, e in enumerate(ev):
    a = e["action"]
    if a == "segment_alloc": gres += e["size"]
    elif a == "segment_free": gres -= e["size"]
    elif a == "alloc":
        na += 1
        live[e["addr"]] = sim.malloc(dev, st.setdefault(e.get("stream", 0), Stream(index=e.get("stream", 0))), int(e["size"]))
        if first is None and sim._trace.max_segment_changes[-1] != gres:
            first = {"event_index": i, "alloc_no": na, "gpu_MiB": gres / MiB, "xmem_MiB": sim._trace.max_segment_changes[-1] / MiB}
    elif a == "free_completed":
        b = live.pop(e["addr"], None)
        if b is not None: sim.free(b)
b = {x["name"]: x for x in T["history"]["boundaries"]}
print(json.dumps({"condition": T["condition"]["condition_id"], "gpu_final_MiB": gres / MiB,
                  "xmem_final_MiB": sim._trace.max_segment_changes[-1] / MiB, "xmem_max_MiB": max(sim._trace.max_segment_changes) / MiB,
                  "gpu_peak_MiB": T["measures"][0]["peak"]["reserved"] / MiB, "first_divergence": first,
                  "segment_size_mismatches": mism[:5], "segments_gpu": len(gseg)}))
