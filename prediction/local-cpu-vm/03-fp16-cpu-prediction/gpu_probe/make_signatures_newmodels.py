"""Conv signatures of the generalisation-test vision models (2026-11-23), only those NOT already in
gpu_probe/conv_signatures.json (3,253 probed on RTX 3090 v3). Same key / format as make_signatures.py (unchanged).
usage (in the CPU image): python gpu_probe/make_signatures_newmodels.py runs/layer_labels_newmodels gpu_probe/conv_signatures_newmodels.json"""
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, "/exp/src")
from cgf680 import replay4  # noqa: E402

KEYS = ("N", "C", "H", "W", "K", "R", "stride", "pad", "dil", "groups")
old = {tuple(s[k] for k in KEYS) for s in json.load(open("/exp/gpu_probe/conv_signatures.json"))["signatures"]}
sigs, seen, per_case = [], set(), {}
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.npz"))):
    cid = os.path.basename(f)[:-4]
    lab = replay4.load_labels(f)
    n_all = n_new = 0
    for c, cp in lab["conv"].items():
        n, cin, h, w = cp["input"]
        cout, _, kh, kw = cp["weight"]
        assert kh == kw and cp["stride"][0] == cp["stride"][-1] and cp["dilation"][0] == cp["dilation"][-1], (cid, cp)
        pad = cp["padding"]
        if pad == "valid":
            pad = [0, 0]
        elif pad == "same":   # odd kernel, stride 1: symmetric (k-1)/2*dil (other cases asymmetric -> stop)
            assert cp["stride"][0] == 1 and kh % 2 == 1, (cid, cp)
            pad = [(kh - 1) // 2 * cp["dilation"][0]] * 2
        assert pad[0] == pad[-1], (cid, cp)
        key = (n, cin, h, w, cout, kh, cp["stride"][0], pad[0], cp["dilation"][0], cp["groups"])
        n_all += 1
        if key in old:
            continue
        n_new += 1
        if key not in seen:
            seen.add(key)
            sigs.append(dict(zip(KEYS, key), src="model:" + cid.split("__")[0]))
    per_case[cid] = {"conv_calls": n_all, "not_in_table": n_new}
json.dump({"note": __doc__, "signatures": sigs, "per_case": per_case}, open(sys.argv[2], "w"))
print("NEW_SIGS", len(sigs), dict(Counter(s["src"] for s in sigs)), "cases", len(per_case),
      "cases_fully_covered", sum(v["not_in_table"] == 0 for v in per_case.values()))
