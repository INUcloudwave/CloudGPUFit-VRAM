"""GPU-only short-lived requests in LLM / T5 / vision traces (2026-11-23): are the two non-cuDNN GPU library requests
found on CNNs (cuBLASLt 1 MiB after linear, max_pool2d int64 indices) also present in other workloads?
A 'transient' = an allocation freed before the next allocation. GPU: RTX 3090 trace, measure_0 window. CPU: the stored
FP16 request stream of the same condition (one measured invocation). Report: GPU transient size multiset minus CPU's.
usage (image): python gpu_only_transients.py <out.json> <trace dir>:<cpu case dir> ..."""
import collections, gzip, json, pickle, sys
from pathlib import Path
import numpy as np


def transients(stream):
    c, last = collections.Counter(), None
    for a, s, k in stream:
        if a == 1:
            last = (s, k)
        elif last is not None and k == last[1]:
            c[last[0]] += 1
            last = None
        else:
            last = None
    return c


def gpu_stream(tdir):
    T = json.loads((tdir / "trace.json").read_text())
    ev = pickle.load(gzip.open(tdir / T["history"]["snapshot_file"], "rb"))["device_traces"][0]
    bounds = T["history"]["boundaries"]
    st, live, na, nf, marks, bi, nid = [], {}, 0, 0, {}, 0, 0

    def chk():
        nonlocal bi
        while bi < len(bounds) and na == bounds[bi]["allocation.all.allocated"] and nf == bounds[bi]["allocation.all.freed"]:
            marks.setdefault(bounds[bi]["name"], len(st)); bi += 1
    chk()
    for e in ev:
        if e["action"] == "alloc":
            live[e["addr"]] = nid; st.append((1, int(e["size"]), nid)); nid += 1; na += 1
        elif e["action"] == "free_completed":
            st.append((0, int(e["size"]), live.pop(e["addr"], None))); nf += 1
        else:
            continue
        chk()
    return st[marks["measure_0_begin"]:marks["measure_0_end"]]


def cpu_stream(cdir):
    z = np.load(cdir / "invocation_fp16_requests.npz")
    aid = z["alloc_id"] if "alloc_id" in z.files else z["id"]
    return [(int(a), int(s), int(k)) for a, s, k in zip(z["action"], z["size"], aid)]


out = {}
for arg in sys.argv[2:]:
    t, c = arg.split(":")
    g, p = gpu_stream(Path(t)), cpu_stream(Path(c))
    tg, tc = transients(g), transients(p)
    extra = {s: n for s, n in (tg - tc).items()}
    out[Path(t).name] = {"gpu_events": len(g), "cpu_events": len(p), "gpu_only_transients": dict(sorted(extra.items())),
                         "cpu_only_transients": dict(sorted((tc - tg).items()))}
    print(Path(t).name, "GPU-only transients (size: count):", {f"{s / 2**20:.3g}MiB": n for s, n in sorted(extra.items())})
json.dump(out, open(sys.argv[1], "w"), indent=1)
