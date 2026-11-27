"""Hybrid v3 (order-preserving): GPU library layer = GPU allocations that are not part of the in-order subsequence
matching the CPU model-layer allocations (greedy by size). Hybrid stream = CPU model-layer events in CPU order, with
each GPU library event inserted right after the CPU counterpart of the GPU event that preceded it. Only the library
layer comes from the GPU; everything else is CPU."""
import sys, json
import numpy as np
OUT = sys.argv[1]; CIDS = sys.argv[2:]
sys.argv = [sys.argv[0], "/dev/null"]
exec(open("/out/verify_guide.py").read().split("\ndef main():")[0])
EFF = 24979374080
import os
CPU_ONLY = os.environ.get("CPU_ONLY") == "1"   # keep only the cuBLAS workspace (known without a GPU)
RUNS = ["warmup_0", "warmup_1", "warmup_2", "measure_0"]
for cid in CIDS:
    gs, gm, gpu = gpu_stream(Path("/f/rtx3090__trace_models/" + cid))
    z = np.load(f"/runs/confirm_hybrid/{cid}_cpu2.npz")
    a, s, k, lay = z["action"], z["size"], z["alloc_id"], z["layer"]
    alay = {int(k[i]): int(lay[i]) for i in range(len(a)) if a[i] == 1}
    cm = [(int(a[i]), int(s[i]), int(k[i])) for i in range(len(a)) if alay.get(int(k[i]), 0) == 0]
    cpos_alloc = {kk: j for j, (x, sz, kk) in enumerate(cm) if x}
    cpos_free = {kk: j for j, (x, sz, kk) in enumerate(cm) if not x}
    calloc = [(sz, kk) for x, sz, kk in cm if x]
    load = [(int(x), int(y), ("L", int(w))) for x, y, w in z["load_events"]]
    stream, marks = list(load), {"model_load_begin": 0, "model_load_end": len(load)}
    rep = {"cid": cid, "gpu_MiB": gpu["m0_start_reserved"] / MiB, "runs": {}}
    for run in RUNS:
        w = window(gs, gm, run)
        def match(skip_transient):
            kmap, ci, lib = {}, 0, set()
            for t, (x, sz, gk) in enumerate(w):   # greedy in-order subsequence match on allocations
                if not x:
                    continue
                transient = t + 1 < len(w) and not w[t + 1][0] and w[t + 1][2] == gk
                if not (skip_transient and transient) and ci < len(calloc) and calloc[ci][0] == sz:
                    kmap[gk] = calloc[ci][1]; ci += 1
                else:
                    lib.add(gk)
            return kmap, ci, lib
        kmap, ci, lib = match(True)               # alloc freed by the very next event = library workspace first
        rule = "transient_first"
        if ci != len(calloc):
            kmap, ci, lib = match(False); rule = "plain"
        subseq = ci == len(calloc)
        ins = {}                                  # CPU position -> GPU library events inserted after it
        last = -1
        for x, sz, gk in w:
            if gk in lib and (not CPU_ONLY or sz == 8519680):
                ins.setdefault(last, []).append((x, sz, ("G", run, gk)))
            elif gk in kmap:
                ck = kmap[gk]
                last = cpos_alloc[ck] if x else cpos_free.get(ck, last)
        marks[run + "_begin"] = len(stream)
        for ev in ins.get(-1, []):
            stream.append(ev)
        for j, (x, sz, kk) in enumerate(cm):
            stream.append((x, sz, ("C", run, kk)))
            for ev in ins.get(j, []):
                stream.append(ev)
        marks[run + "_end"] = len(stream)
        rep["runs"][run] = {"rule": rule, "cpu_model_is_ordered_subsequence_of_gpu": subseq, "matched": ci, "cpu_model_allocs": len(calloc),
                            "gpu_library_allocs": sum(1 for x, sz, gk in w if x and gk in lib),
                            "gpu_library_MiB": round(sum(sz for x, sz, gk in w if x and gk in lib) / MiB, 1)}
    r = replay(stream, marks, EFF, "descending")
    rep["hybrid_MiB"] = None if r["oom"] else r["m0_start_reserved"] / MiB
    print(json.dumps(rep), flush=True)
    open(OUT, "a").write(json.dumps(rep) + "\n")
