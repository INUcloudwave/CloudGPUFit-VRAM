#!/usr/bin/env python3
"""Are the 4 invocations' request sequences identical? Compare per-region (alloc sizes in order, and the
alloc/free interleaving with free sizes) for every case directory given."""
import sys, json
import numpy as np
for d in sys.argv[1:]:
    z = np.load(f"{d}/events.npz")
    act, size, reg, aid = z["action"], z["size"], z["region"], z["alloc_id"]
    seqs = []
    for r in range(4):
        m = reg == r
        seqs.append((act[m], size[m]))
    same = [bool(len(seqs[0][0]) == len(s[0]) and (seqs[0][0] == s[0]).all() and (seqs[0][1] == s[1]).all()) for s in seqs[1:]]
    same23 = [bool(len(seqs[1][0]) == len(s[0]) and (seqs[1][0] == s[0]).all() and (seqs[1][1] == s[1]).all()) for s in seqs[2:]]
    print(json.dumps({"case": d.split("/")[-1], "events_per_run": [int(len(s[0])) for s in seqs],
                      "run0_vs_1_2_3_identical": same, "run1_vs_2_3_identical": same23}))
