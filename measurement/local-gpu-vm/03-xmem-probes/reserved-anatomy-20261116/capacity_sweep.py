#!/usr/bin/env python3
"""R_inf / R_min of one GPU request stream with the xMem allocator (CPU only).
  CUDA_VISIBLE_DEVICES= python capacity_sweep.py <xmem_repo> <trace.json> <out.json>
R_inf: reserved at measure_0 start with effectively unlimited capacity (no OOM-retry can happen).
R_min: smallest capacity (2 MiB resolution, binary search) for which the whole stream replays without OOM.
Also reports reserved at measure_0 start when capacity = the RTX 3090 cuda_total_memory (what the GPU actually had)."""
import gzip, json, logging, pickle, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1]); logging.disable(logging.CRITICAL)
from perf_estimator.allocator.allocator import CachingAllocator, Device, Stream
T = json.loads(Path(sys.argv[2]).read_text()); MiB = 2**20
ev = pickle.load(gzip.open(Path(sys.argv[2]).with_name(T["history"]["snapshot_file"]), "rb"))["device_traces"][0]
b = {x["name"]: x for x in T["history"]["boundaries"]}
m0 = (b["measure_0_begin"]["allocation.all.allocated"], b["measure_0_begin"]["allocation.all.freed"])
def run(cap):
    sim = CachingAllocator(allowed_memory_maximum=cap); dev, st, live = Device(index=0), {}, {}
    na = nf = 0; r_m0 = None
    try:
        for e in ev:
            if e["action"] == "alloc":
                live[e["addr"]] = sim.malloc(dev, st.setdefault(e.get("stream", 0), Stream(index=e.get("stream", 0))), int(e["size"])); na += 1
            elif e["action"] == "free_completed":
                x = live.pop(e["addr"], None)
                if x is not None: sim.free(x)
                nf += 1
            if r_m0 is None and (na, nf) == m0: r_m0 = sim._trace.max_segment_changes[-1]
    except MemoryError:
        return None
    return {"reserved_at_measure0_MiB": r_m0 / MiB, "max_reserved_MiB": max(sim._trace.max_segment_changes) / MiB}
gpu_cap = T["environment"]["cuda_total_memory"]
r_inf = run(1 << 40)
peak_alloc = T["measures"][0]["peak"]["allocated"]
lo, hi = peak_alloc // MiB, int(r_inf["max_reserved_MiB"]) + 2
while hi - lo > 2:
    mid = (lo + hi) // 2
    if run(mid * MiB) is None: lo = mid
    else: hi = mid
out = {"condition_id": T["condition"]["condition_id"],
       "gpu_measure0_peak_reserved_MiB": T["measures"][0]["peak"]["reserved"] / MiB,
       "gpu_measure0_peak_allocated_MiB": round(peak_alloc / MiB, 1),
       "xmem_R_inf": r_inf, "xmem_at_rtx3090_capacity": run(gpu_cap), "xmem_R_min_capacity_MiB": hi,
       "xmem_at_R_min": run(hi * MiB)}
Path(sys.argv[3]).write_text(json.dumps(out, indent=1)); print(json.dumps(out))
