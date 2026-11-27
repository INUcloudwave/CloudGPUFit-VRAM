"""Family + grid rule for cuDNN workspace (2026-11-23): predict candidate-1 workspace of a conv shape from the generic
grid shapes only (src grid / grid_dw of the RTX 3090 v3 probe): same family (kernel, stride, groups type), nearest grid
shape by log distance over N, C, H, K, workspace = (grid ws / grid output bytes) x output bytes.
  python grid_rule.py selftest  -> predicts the 1,180 model shapes of the v3 probe (known truth)
  python grid_rule.py predict <signatures.json> <out.jsonl>  -> pre-registered predictions for unseen shapes"""
import json
import math
import sys
from collections import Counter

PROBE = "gpu_probe/results/rtx3090_v3/probe_rtx3090.jsonl"
M = 1 << 20


def fam(r):
    g = r["groups"]
    return (r["R"], r["stride"], "dw" if g > 1 and g == r["C"] else ("g" if g > 1 else "1"))


def out_bytes(s):
    ho = (s["H"] + 2 * s["pad"] - s["dil"] * (s["R"] - 1) - 1) // s["stride"] + 1
    wo = (s["W"] + 2 * s["pad"] - s["dil"] * (s["R"] - 1) - 1) // s["stride"] + 1
    return 2 * s["N"] * s["K"] * ho * wo


rows = [json.loads(x) for x in open(PROBE) if '"N"' in x]
rows = [r for r in rows if r.get("status") == "ok"]
grid = [r for r in rows if r["src"].startswith("grid")]


def predict(s):
    cand = [g for g in grid if fam(g) == fam(s)]
    if not cand:
        return None, None
    g = min(cand, key=lambda g: sum(abs(math.log(g[k]) - math.log(s[k])) for k in ("N", "C", "H", "K")))
    o = g["candidates"][0]["out_bytes"]
    return (g["candidates"][0]["ws"] / o * out_bytes(s) if o else 0.0), g


if sys.argv[1] == "selftest":
    c = Counter()
    for m in (r for r in rows if r["src"].startswith("model")):
        p, _ = predict(m)
        if p is None:
            c["no_family"] += 1
            continue
        e = abs(p - m["candidates"][0]["ws"])
        c["<=1MiB" if e <= M else "<=32MiB" if e <= 32 * M else ">32MiB"] += 1
    print(dict(c))
else:
    sigs = json.load(open(sys.argv[2]))["signatures"]
    c = Counter()
    with open(sys.argv[3], "w") as f:
        for s in sigs:
            p, g = predict(s)
            c[("no_family", fam(s)) if p is None else "predicted"] += 1
            f.write(json.dumps({**s, "fam": list(fam(s)), "ws_pred": None if p is None else int(p),
                                "nearest_grid": None if g is None else {k: g[k] for k in ("N", "C", "H", "K")}}) + "\n")
    print(dict(c))
