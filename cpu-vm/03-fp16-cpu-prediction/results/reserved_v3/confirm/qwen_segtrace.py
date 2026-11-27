"""Confirmation (qwen): in the v3 replay (descending + workspace, effective capacity), list every segment creation
(run, index in invocation, request size, segment size) and the reserved at measure_0 start, to locate where a GPU-only
extra segment of T*8960*2 B would have to appear. usage: segtrace.py <case_dir> <ws_rel> <out.json>"""
import json, sys
from cgf680 import replay3 as R
cdir, ws_rel, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
st = R.load_stream(cdir)
sim, dev, stream = R.new_allocator(24979374080, "descending")
M = 1 << 20
def res(): return sim.total_allocated_memory
live, segs = {}, []
for a, s, k in st["load"]:
    if a: live[("L", k)] = sim.malloc(dev, stream, s)
    else:
        b = live.pop(("L", k), None)
        if b is not None: sim.free(b)
info = {"after_load_MiB": res() / M}
for run in R.RUN_NAMES:
    if run == "measure_0": info["m0_start_MiB"] = res() / M
    lv = {}
    for j, (a, s, k) in enumerate(zip(st["act"], st["size"], st["aid"])):
        if run == "warmup_0" and j == ws_rel:
            r0 = res(); sim.malloc(dev, stream, R.CUBLAS_WS)
            if res() > r0: segs.append([run, j, "WS", R.CUBLAS_WS / M, (res() - r0) / M])
        if a:
            r0 = res(); lv[k] = sim.malloc(dev, stream, s)
            if res() > r0: segs.append([run, j, s, round(s / M, 2), (res() - r0) / M])
        else:
            b = lv.pop(k, None)
            if b is not None: sim.free(b)
info["n_invocation_events"] = len(st["act"])
info["segment_creations"] = [x for x in segs if x[4] >= 8]
info["small_segment_creations"] = len([x for x in segs if x[4] < 8])
json.dump(info, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in info.items() if k != "segment_creations"}), len(info["segment_creations"]))
for x in info["segment_creations"]: print(x)
