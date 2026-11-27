"""qwen: compare GPU and CPU request streams on big requests (>= 1 MiB), GPU-only / CPU-only sizes removed, warmup_0.
Prints the first position where the (action, size) sequences differ, with context. usage: align_big.py <cid>=<gdir>=<cdir>"""
import sys
exec(open("/out/verify_guide.py").read().split("\ndef main():")[0])
GPU_ONLY = {1048576, 1572864, 8519680}
CPU_ONLY = {6574080}


def seq(win, drop):
    live, out = {}, []
    for a, s, k in win:
        if a:
            live[k] = s
            if s >= MiB and s not in drop:
                out.append(("A", s, k))
        else:
            s0 = live.pop(k, s)
            if s0 >= MiB and s0 not in drop:
                out.append(("F", s0, k))
    return out


for spec in sys.argv[1:]:
    cid, gdir, cdir = spec.split("=")
    gs, gm, gpu = gpu_stream(Path(gdir))
    cs, cm, _ = cpu_stream(Path(cdir))
    for run in ("warmup_0", "measure_0"):
        g = seq(window(gs, gm, run), GPU_ONLY)
        c = seq(window(cs, cm, run), CPU_ONLY)
        i = 0
        while i < min(len(g), len(c)) and g[i][:2] == c[i][:2]:
            i += 1
        print(json.dumps({"cid": cid, "run": run, "n_gpu": len(g), "n_cpu": len(c), "first_mismatch": i,
                          "gpu_ctx": [(a, s) for a, s, k in g[max(0, i - 4):i + 6]],
                          "cpu_ctx": [(a, s) for a, s, k in c[max(0, i - 4):i + 6]]}), flush=True)
