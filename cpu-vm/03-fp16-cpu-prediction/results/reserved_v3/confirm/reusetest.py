"""Retry conditions with GPU traces: replay the GPU's own request stream with VA policies desc / reuse_top / reuse_bottom.
Confirms whether freed-range reuse by the driver explains the remaining retry error."""
import sys, json
OUT = sys.argv[1]; SPECS = sys.argv[2:]
sys.argv = [sys.argv[0], "/dev/null"]
src = open("/out/verify_guide.py").read().split("\ndef main():")[0]
exec(src)
exec(open("/out/addrmodel_va.py").read())
_orig = make_sim
def make_sim(cap, policy):
    if policy in ("reuse_top", "reuse_bottom"):
        sim = CachingAllocator(allowed_memory_maximum=cap)
        sim._trace = LightTrace(sim._segments)
        va = VA(policy)
        sim._gpu_device.malloc = lambda size: Fake(va.alloc(size))
        sim._gpu_device.addr_free = lambda addr: va.free(addr)
        return sim
    return _orig(cap, policy)
EFF = 24979374080
for spec in SPECS:
    cid, gdir = spec.split("=")
    gs, gm, gpu = gpu_stream(Path(gdir))
    for pol in ("descending", "reuse_top", "reuse_bottom"):
        r = replay(gs, gm, EFF, pol)
        row = {"cid": cid, "policy": pol, "gpu_m0_start_reserved": gpu["m0_start_reserved"],
               "pred": None if r["oom"] else r["m0_start_reserved"]}
        row["err_MiB"] = None if row["pred"] is None else (row["pred"] - gpu["m0_start_reserved"]) / MiB
        print(json.dumps(row), flush=True)
        open(OUT, "a").write(json.dumps(row) + "\n")
