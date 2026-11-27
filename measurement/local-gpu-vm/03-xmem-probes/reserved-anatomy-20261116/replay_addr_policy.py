#!/usr/bin/env python3
"""Replay GPU request stream with a synthetic segment-address policy (no GPU addresses used).
  CUDA_VISIBLE_DEVICES= python replay_addr_policy.py <xmem_repo> <policy> <trace.json>...
policies: xmem (stock), descending (each new segment below all previous), ascending."""
import gzip, json, logging, pickle, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1]); logging.disable(logging.CRITICAL)
from perf_estimator.allocator.allocator import CachingAllocator, Device, Stream
pol = sys.argv[2]; MiB = 2**20
class Fake:
    def __init__(self, a): self.address = a
for tj in sys.argv[3:]:
    T = json.loads(Path(tj).read_text())
    ev = pickle.load(gzip.open(Path(tj).with_name(T["history"]["snapshot_file"]), "rb"))["device_traces"][0]
    sim = CachingAllocator(allowed_memory_maximum=T["environment"]["cuda_total_memory"])
    if pol != "xmem":
        state = {"lo": 1 << 46, "hi": 1 << 40}
        def gm(size, state=state):
            if pol == "descending":
                state["lo"] -= size; return Fake(state["lo"])
            a = state["hi"]; state["hi"] += size; return Fake(a)
        sim._gpu_device.malloc = gm
        sim._gpu_device.addr_free = lambda addr: None
    dev, st, live = Device(index=0), {}, {}
    b = {x["name"]: x for x in T["history"]["boundaries"]}; m0 = (b["measure_0_begin"]["allocation.all.allocated"], b["measure_0_begin"]["allocation.all.freed"])
    na = nf = 0; r = None
    for e in ev:
        if e["action"] == "alloc":
            live[e["addr"]] = sim.malloc(dev, st.setdefault(e.get("stream", 0), Stream(index=e.get("stream", 0))), int(e["size"])); na += 1
        elif e["action"] == "free_completed":
            x = live.pop(e["addr"], None)
            if x is not None: sim.free(x)
            nf += 1
        if r is None and (na, nf) == m0: r = sim._trace.max_segment_changes[-1]
    print(json.dumps({"policy": pol, "condition": T["condition"]["condition_id"], "gpu_MiB": b["measure_0_begin"]["reserved_bytes.all.current"] / MiB, "xmem_MiB": r / MiB}))
