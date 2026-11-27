"""CPU stream + GPU workspace injections (positions matched from the GPU trace). usage: wstest.py <out.jsonl> <spec> [caps...]"""
import sys, json
OUT = sys.argv[1]; SPEC = sys.argv[2]; CAPS = [int(x) for x in sys.argv[3:]]
sys.argv = [sys.argv[0], "/dev/null"]
exec(open("/out/verify_guide.py").read().split("\ndef main():")[0])
cid, gdir, cdir = SPEC.split("=")
gs, gm, gpu = gpu_stream(Path(gdir)); cs, cm, _ = cpu_stream(Path(cdir))
gw = [x for x in window(gs, gm, "warmup_0") if x[0]]
del gs
def matched(target):
    """position in CPU stream right after the CPU request matching the GPU request that precedes `target` in warmup_0"""
    for gi, (_, s, _) in enumerate(gw):
        if s == target: break
    else: return None, None
    WS = (1048576, 8519680)   # GPU-only workspace sizes: anchor on the previous request that the CPU stream also has
    pj = gi - 1
    while pj >= 0 and gw[pj][1] in WS: pj -= 1
    if pj < 0: return cm["warmup_0_begin"], gi
    sp = gw[pj][1]; c = sum(1 for x in gw[:pj + 1] if x[1] == sp)
    n = 0
    for i in range(cm["warmup_0_begin"], cm["warmup_0_end"]):
        a, s, k = cs[i]
        if a and s == sp:
            n += 1
            if n == c: return i + 1, gi
    return None, gi
def replay2(stream, marks, cap, inj):
    # inj: list of (pos, size, transient)
    ext = []; m2 = dict(marks); add = sorted(inj)
    j = 0; shift = {}
    out = []
    for i, ev in enumerate(stream):
        while j < len(add) and add[j][0] == i:
            p, s, tr = add[j]; key = ("INJ", j)
            out.append((1, s, key))
            if tr: out.append((0, s, key))
            j += 1
        shift[i] = len(out); out.append(ev)
    shift[len(stream)] = len(out)
    m2 = {n: shift[p] for n, p in marks.items()}
    return replay(out, m2, cap, "descending")
pws, gws = matched(8519680); plt, glt = matched(1048576)
row = {"cid": cid, "gpu_m0": gpu["m0_start_reserved"], "gpu_ws_alloc_idx": gws, "cpu_ws_pos": pws, "gpu_lt_alloc_idx": glt, "cpu_lt_pos": plt}
caps = CAPS or [gpu["cap"]]
V = {"desc": [], "ws_matched": [(pws, 8519680, False)] if pws is not None else None,
     "ws_w0start": [(cm["warmup_0_begin"], 8519680, False)],
     "ws_matched+lt1": ([(pws, 8519680, False)] + ([(plt, 1048576, True)] if plt is not None else [])) if pws is not None else None}
for cap in caps:
    for name, inj in V.items():
        if inj is None: continue
        r = replay2(cs, cm, cap, inj)
        row[f"{name}@{cap}"] = None if r["oom"] else r["m0_start_reserved"]
        print(cid, cap, name, "OOM" if r["oom"] else (r["m0_start_reserved"] - gpu["m0_start_reserved"]) / MiB, flush=True)
open(OUT, "a").write(json.dumps(row) + "\n")
