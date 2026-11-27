#!/usr/bin/env python3
"""Request-level comparison: GPU measure_0 requests (INUGPU trace) vs CPU single-invocation FP16-sized requests.
Reports counts/bytes, multiset differences (GPU-only / CPU-only sizes) and the largest contributors."""
import sys, json
from collections import Counter
import numpy as np
sys.path.insert(0, "/diag/src")
from cgf_diag.events import load_snapshot
H = "/handoff/reference/traces_attn_default"
out = []
for cid in sys.argv[1:]:
    t = json.load(open(f"{H}/{cid}/trace.json")); b = {x["name"]: x for x in t["history"]["boundaries"]}
    ev = load_snapshot(f"{H}/{cid}/trace.snapshot.pickle.gz")["device_traces"][0]
    s0, s1 = b["measure_0_begin"]["allocation.all.allocated"], b["measure_0_end"]["allocation.all.allocated"]
    g, a = [], 0
    for e in ev:
        if e["action"] == "alloc":
            if s0 <= a < s1:
                g.append(e["size"])
            a += 1
    z = np.load(f"/exp/runs/main395/cases/{cid}/invocation_fp16_requests.npz")
    c = [int(x) for x in z["size"][z["action"] == 1]]
    G, C = Counter(g), Counter(c)
    gonly, conly = G - C, C - G
    top = lambda d: sorted(((s * n, s, n) for s, n in d.items()), reverse=True)[:8]
    r = {"cid": cid, "gpu_requests": len(g), "cpu_requests": len(c), "gpu_bytes": sum(g), "cpu_bytes": sum(c),
         "same_multiset": G == C, "gpu_only_n": sum(gonly.values()), "gpu_only_bytes": sum(s * n for s, n in gonly.items()),
         "cpu_only_n": sum(conly.values()), "cpu_only_bytes": sum(s * n for s, n in conly.items()),
         "gpu_only_top(bytes,size,count)": top(gonly), "cpu_only_top(bytes,size,count)": top(conly)}
    out.append(r)
    print(json.dumps(r))
