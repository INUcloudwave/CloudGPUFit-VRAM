"""Hybrid check, analysis side.
CPU model layer (layer 0, from cpu_layers.py) + GPU library layer (GPU allocations not matched to any CPU model-layer
allocation) -> replay (descending, effective capacity) and compare with the GPU measurement.
Also: are the two streams identical once each side's library layer is removed?"""
import sys, json, difflib
from collections import Counter, defaultdict
import numpy as np
OUT = sys.argv[1]; CIDS = sys.argv[2:]
sys.argv = [sys.argv[0], "/dev/null"]
exec(open("/out/verify_guide.py").read().split("\ndef main():")[0])
EFF = 24979374080
RUNS = ["warmup_0", "warmup_1", "warmup_2", "measure_0"]
for cid in CIDS:
    gs, gm, gpu = gpu_stream(Path("/f/rtx3090__trace_models/" + cid))
    z = np.load(f"/runs/confirm_hybrid/{cid}_cpu.npz")
    a, s, k, lay, fn = z["action"], z["size"], z["alloc_id"], z["layer"], z["func"]
    names = list(z["func_names"])
    alay = {int(k[i]): int(lay[i]) for i in range(len(a)) if a[i] == 1}
    afn = {int(k[i]): names[int(fn[i])] if fn[i] >= 0 else "?" for i in range(len(a)) if a[i] == 1}
    cpu_red = [(int(a[i]), int(s[i]), int(k[i])) for i in range(len(a)) if alay.get(int(k[i]), 0) == 0]
    cpu_lib_bytes = sum(int(s[i]) for i in range(len(a)) if a[i] == 1 and alay.get(int(k[i])) == 1)
    load = [(int(x), int(y), ("L", int(w))) for x, y, w in z["load_events"]]
    stream, marks = list(load), {"model_load_begin": 0, "model_load_end": len(load)}
    rep = {"cid": cid, "gpu_m0_start_reserved_MiB": gpu["m0_start_reserved"] / MiB, "runs": {}}
    gpu_lib_all = []
    for run in RUNS:
        w = window(gs, gm, run)
        g_alloc = [(i, sz) for i, (x, sz, kk) in enumerate(w) if x]
        c_alloc = [sz for x, sz, kk in cpu_red if x]
        sm = difflib.SequenceMatcher(None, [sz for _, sz in g_alloc], c_alloc, autojunk=False)
        matched = set()
        cpu_unmatched = 0
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                matched.update(g_alloc[t][0] for t in range(i1, i2))
            else:
                cpu_unmatched += (j2 - j1)
        lib_keys = {w[i][2] for i, _ in g_alloc if i not in matched}
        g_red = [(x, sz) for x, sz, kk in w if kk not in lib_keys]
        same = [(x, sz) for x, sz, kk in cpu_red] == g_red
        # hybrid (full-event alignment): tokens (alloc/free, size); equal blocks -> CPU events, GPU-only -> GPU events,
        # CPU-only -> CPU events. Matched allocs map GPU key -> CPU key so GPU-side frees free the right block.
        gt = [(x, sz) for x, sz, kk in w]
        ct = [(x, sz) for x, sz, kk in cpu_red]
        sm2 = difflib.SequenceMatcher(None, gt, ct, autojunk=False)
        marks[run + "_begin"] = len(stream)
        kmap, lib_list, freed = {}, [], set()
        gonly = conly = 0
        def emit(ev):
            x, sz, key = ev
            if not x:
                if key in freed:
                    return
                freed.add(key)
            stream.append(ev)
        for tag, i1, i2, j1, j2 in sm2.get_opcodes():
            if tag == "equal":
                for t in range(i2 - i1):
                    (gx, gsz, gk), (cx, csz, ck) = w[i1 + t], cpu_red[j1 + t]
                    if gx:
                        kmap[gk] = ("C", run, ck)
                    emit((cx, csz, ("C", run, ck)))
                continue
            for t in range(i1, i2):
                gx, gsz, gk = w[t]
                key = kmap.get(gk, ("G", run, gk))
                emit((gx, gsz, key)); gonly += 1
                if gx:
                    lib_list.append((gsz, None))
            for t in range(j1, j2):
                cx, csz, ck = cpu_red[t]
                emit((cx, csz, ("C", run, ck))); conly += 1
        rep_extra = {"gpu_only_events": gonly, "cpu_only_events": conly}
        marks[run + "_end"] = len(stream)
        gpu_lib_all += [(run,) + t for t in lib_list]
        rep["runs"][run] = {"gpu_lib_allocs": len(lib_list), "gpu_lib_MiB": round(sum(t[0] for t in lib_list) / MiB, 1),
                            "cpu_model_allocs_unmatched": cpu_unmatched, "reduced_streams_identical": same, **rep_extra}
    marks["measure_0_begin"] = marks["measure_0_begin"]
    r = replay(stream, marks, EFF, "descending")
    rep["hybrid_replay_MiB"] = None if r["oom"] else r["m0_start_reserved"] / MiB
    rep["cpu_library_layer_MiB_per_run"] = round(cpu_lib_bytes / MiB, 1)
    w0 = [t for t in gpu_lib_all if t[0] == "warmup_0"]
    by = defaultdict(lambda: [0, 0])
    for _, sz, f in w0:
        by[f][0] += 1; by[f][1] += sz
    rep["gpu_lib_warmup0_by_next_op"] = {f: [c, round(b / MiB, 1)] for f, (c, b) in by.items()}
    rep["gpu_lib_warmup0_sizes_MiB"] = [round(sz / MiB, 3) for _, sz, f in w0][:40]
    print(json.dumps(rep), flush=True)
    open(OUT, "a").write(json.dumps(rep) + "\n")
