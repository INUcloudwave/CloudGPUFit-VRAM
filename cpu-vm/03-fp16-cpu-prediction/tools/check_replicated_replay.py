#!/usr/bin/env python3
"""Replay 'invocation 0 replicated 4x' and compare with the replay of the real 4-invocation trace.
If identical for a case, tracing one invocation is sufficient for that case."""
import sys, json
import numpy as np
from cgf680.cpu_worker import replay, measure_stats
for d in sys.argv[1:]:
    z = np.load(f"{d}/events.npz")
    real = json.load(open(f"{d}/result.json"))["prediction"]
    mp = [int(x) for x in z["marker_pos"]]
    a, s, i, r = z["action"], z["size"], z["alloc_id"], z["region"]
    seg = slice(mp[0], mp[1])                       # run0 + its output release
    run_len = int((r[seg] == 0).sum())             # events inside region 0
    n = mp[1] - mp[0]
    big = int(i.max()) + 1
    A = np.concatenate([a[seg]] * 4); S = np.concatenate([s[seg]] * 4)
    I = np.concatenate([i[seg] + k * big for k in range(4)]); R = np.concatenate([r[seg]] * 4)
    marks = {0: 0, 1: n, 2: 2 * n, 3: 3 * n, 4: 3 * n + run_len, 5: 4 * n}
    _, snap, sg = replay({"action": A, "size": S, "alloc_id": I, "region": R}, marks,
                         [int(x) for x in z["resident_sizes"]], ["r0"])
    syn = measure_stats(snap, sg)
    keys = [k for k in real if k.endswith("_bytes")]
    print(json.dumps({"case": d.split("/")[-1], "identical": all(syn[k] == real[k] for k in keys),
                      "diff": {k: syn[k] - real[k] for k in keys if syn[k] != real[k]}}))
