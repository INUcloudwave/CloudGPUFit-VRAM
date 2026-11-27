"""Retry conditions: GPU stream & CPU stream(+ws matched), descending policy; caps cuda_total / effective / 1TiB; R_min (2 MiB bisection)."""
import sys, json
OUT, SPEC = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0], "/dev/null"]
exec(open("/out/verify_guide.py").read().split("\ndef main():")[0])
src = open("/out/wstest.py").read()
exec(src[src.index("def matched"):src.index("pws, gws = matched")])
cid, gdir, cdir = SPEC.split("=")
gs, gm, gpu = gpu_stream(Path(gdir)); cs, cm, _ = cpu_stream(Path(cdir))
gw = [x for x in window(gs, gm, "warmup_0") if x[0]]
pws, gws = matched(8519680)
EFF = 24979374080
row = {"cid": cid, "gpu_m0": gpu["m0_start_reserved"], "gpu_peak_alloc": gpu["m0_peak_allocated"], "cpu_ws_pos": pws}
def rmin(fn, lo, hi):
    lo //= MiB; hi = hi // MiB + 2
    while hi - lo > 2:
        mid = (lo + hi) // 2
        if fn(mid * MiB)["oom"]: lo = mid
        else: hi = mid
    return hi * MiB
streams = {"gpu_desc": lambda cap: replay(gs, gm, cap, "descending"),
           "cpu_desc_ws": lambda cap: replay2(cs, cm, cap, [(pws, 8519680, False)])}
for name, fn in streams.items():
    for capn, cap in (("cuda_total", gpu["cap"]), ("effective", EFF), ("inf", 1 << 40)):
        r = fn(cap)
        row[f"{name}@{capn}"] = None if r["oom"] else r["m0_start_reserved"]
        if capn == "inf": row[f"{name}_max_alloc"] = r["max_allocated"]; rinf = r["max_reserved"]
        print(cid, name, capn, "OOM" if r["oom"] else f"{r['m0_start_reserved']/MiB:.0f} MiB (GPU {gpu['m0_start_reserved']/MiB:.0f})", flush=True)
    row[f"{name}_R_min"] = rmin(fn, row[f"{name}_max_alloc"], rinf)
    print(cid, name, "R_min", row[f"{name}_R_min"] / MiB, flush=True)
open(OUT, "a").write(json.dumps(row) + "\n")
