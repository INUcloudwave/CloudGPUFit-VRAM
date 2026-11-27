#!/usr/bin/env python3
"""Find the first event where xMem reserved differs from GPU reserved (replaying the GPU request stream).
  CUDA_VISIBLE_DEVICES= python first_divergence.py <xmem_repo> <trace.json>"""
import gzip, json, logging, pickle, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1]); logging.disable(logging.CRITICAL)
from perf_estimator.allocator.allocator import CachingAllocator, Device, Stream
T = json.loads(Path(sys.argv[2]).read_text())
ev = pickle.load(gzip.open(Path(sys.argv[2]).with_name(T["history"]["snapshot_file"]), "rb"))["device_traces"][0]
sim = CachingAllocator(allowed_memory_maximum=T["environment"]["cuda_total_memory"])
dev, st, live = Device(index=0), {}, {}
gres = 0; MiB = 2**20; nalloc = 0; last_gpu_seg = []
for i, e in enumerate(ev):
    a = e["action"]
    if a == "segment_alloc":
        gres += e["size"]; last_gpu_seg.append(("alloc", e["size"]))
    elif a == "segment_free":
        gres -= e["size"]; last_gpu_seg.append(("free", e["size"]))
    elif a == "alloc":
        nalloc += 1
        live[e["addr"]] = sim.malloc(dev, st.setdefault(e.get("stream", 0), Stream(index=e.get("stream", 0))), int(e["size"]))
        xres = sim._trace.max_segment_changes[-1]
        if xres != gres:
            print(json.dumps({"condition": T["condition"]["condition_id"], "event_index": i, "alloc_no": nalloc,
                              "request_bytes": e["size"], "gpu_reserved_MiB": gres / MiB, "xmem_reserved_MiB": xres / MiB,
                              "gpu_recent_segment_events": last_gpu_seg[-3:],
                              "live_tensors": len(live), "live_bytes_MiB": round(sum(b.size for b in live.values() if b) / MiB, 1)}))
            # show xmem's free large blocks near the request size
            break
    elif a == "free_completed":
        b = live.pop(e["addr"], None)
        if b is not None: sim.free(b)
else:
    print("no divergence")
