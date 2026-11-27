"""qwen: where does the GPU replay get its extra segment? Replay GPU and CPU streams (descending, GPU cap) and
record reserved after every big event (>= 1 MiB, GPU-only/CPU-only sizes not counted; both streams have the same
number of such events). Print the first big-event index where reserved differs by >= 100 MiB, with context."""
import sys
exec(open("/out/verify_guide.py").read().split("\ndef main():")[0])
GPU_ONLY = {1048576, 1572864, 8519680}
CPU_ONLY = {6574080}


def trace_res(stream, marks, cap, drop):
    sim = make_sim(cap, "descending")
    dev, st = Device(index=0), Stream(index=0)
    live, size_of, res, ev, bound = {}, {}, [], [], {}
    inv = {v: k for k, v in marks.items()}
    for i, (a, s, k) in enumerate(stream):
        if i in inv:
            bound[inv[i]] = len(res)
        if a:
            live[k] = sim.malloc(dev, st, s)
            size_of[k] = s
            big = s >= MiB and s not in drop
        else:
            b = live.pop(k, None)
            if b is not None:
                sim.free(b)
            s = size_of.get(k, s)
            big = s >= MiB and s not in drop
        if big:
            res.append(sim.total_allocated_memory)
            ev.append(("A" if a else "F", s))
    return res, ev, bound


spec = sys.argv[1]
cid, gdir, cdir = spec.split("=")
gs, gm, gpu = gpu_stream(Path(gdir))
cs, cm, _ = cpu_stream(Path(cdir))
rg, eg, bg = trace_res(gs, gm, gpu["cap"], GPU_ONLY)
rc, ec, bc = trace_res(cs, cm, gpu["cap"], CPU_ONLY)
n = min(len(rg), len(rc))
j = next((i for i in range(n) if abs(rg[i] - rc[i]) >= 100 * MiB), None)
out = {"cid": cid, "n_gpu": len(rg), "n_cpu": len(rc), "first_div": j, "gpu_bounds": bg, "cpu_bounds": bc,
       "final_gpu_MiB": rg[-1] / MiB, "final_cpu_MiB": rc[-1] / MiB}
if j is not None:
    out["res_gpu_MiB"] = [round(x / MiB) for x in rg[j - 6:j + 3]]
    out["res_cpu_MiB"] = [round(x / MiB) for x in rc[j - 6:j + 3]]
    out["ev_gpu"] = eg[j - 30:j + 3]
    out["ev_cpu"] = ec[j - 30:j + 3]
print(json.dumps(out))
