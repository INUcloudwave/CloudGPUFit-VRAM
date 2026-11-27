"""cuDNN workspace per conv call from RTX 3090 GPU traces (measure_0), mapped to conv parameters via the CPU layer
labels (CPU model layer == GPU model layer in order, verified 3/3 in confirm/hybrid).

GPU library request = GPU allocation not in the in-order subsequence matching the CPU model-layer allocations
(transient allocations, freed by the very next event, are classified as library first: hybrid4 rule).
Each GPU library request is attributed to the CPU call of the model-layer allocation right before it.
usage (inside the image, /f = INUGPU trace copy, /exp = experiment root):
  python wsmap.py <out.jsonl> <case_id> ...
"""
import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/exp/src")
from cgf680 import replay4  # noqa: E402


def gpu_stream(tdir):                               # = guide_verification/verify_guide.py gpu_stream (stream part)
    T = json.loads((tdir / "trace.json").read_text())
    ev = pickle.load(gzip.open(tdir / T["history"]["snapshot_file"], "rb"))["device_traces"][0]
    bounds = T["history"]["boundaries"]
    stream, live, na, nf, marks, bi = [], {}, 0, 0, {}, 0

    def chk():
        nonlocal bi
        while bi < len(bounds) and na == bounds[bi]["allocation.all.allocated"] and nf == bounds[bi]["allocation.all.freed"]:
            marks.setdefault(bounds[bi]["name"], len(stream))
            bi += 1
    chk()
    nid = 0
    for e in ev:
        a = e["action"]
        if a == "alloc":
            live[e["addr"]] = nid
            stream.append((1, int(e["size"]), nid))
            nid += 1
            na += 1
        elif a == "free_completed":
            k = live.pop(e["addr"], None)
            stream.append((0, int(e["size"]), k))
            nf += 1
        else:
            continue
        chk()
    return stream, marks


out = open(sys.argv[1], "w")
for cid in sys.argv[2:]:
    gs, gm = gpu_stream(Path("/f") / cid)
    w = gs[gm["measure_0_begin"]:gm["measure_0_end"]]
    lab = replay4.load_labels(f"/exp/runs/layer_labels/{cid}.npz")
    act, size, aid, layer, call, func = lab["act"], lab["size"], lab["aid"], lab["layer"], lab["call"], lab["func"]
    calloc = [(int(size[i]), int(call[i]), str(func[i])) for i in range(len(act)) if act[i] == 1 and layer[i] == 0]
    ci, prev_call, recs, lib_n = 0, None, [], 0
    for t, (x, sz, gk) in enumerate(w):
        if not x:
            continue
        transient = t + 1 < len(w) and not w[t + 1][0] and w[t + 1][2] == gk
        if not transient and ci < len(calloc) and calloc[ci][0] == sz:
            prev_call = (calloc[ci][1], calloc[ci][2], calloc[ci][0])
            ci += 1
            continue
        lib_n += 1
        c, fn, osz = prev_call if prev_call else (None, None, None)
        recs.append({"case_id": cid, "ws": sz, "transient": transient, "call": c, "func": fn, "out_bytes": osz,
                     "conv": lab["conv"].get(c) if c is not None else None})
    for r in recs:
        out.write(json.dumps(r) + "\n")
    print(json.dumps({"case_id": cid, "gpu_allocs": sum(1 for e in w if e[0]), "matched_model": ci,
                      "cpu_model_allocs": len(calloc), "library": lib_n}), flush=True)
