"""Deviceless (cuDNN 9.28, RTX 3090 properties) vs GPU probe (torch 2.6 + cuDNN 9.28, same GPU), 4,206 shapes.
GPU probe 'ws' = transient bytes during the conv call beyond the output (512-rounded). Rules tested per shape:
  plain : r512(plan ws) == GPU ws
  +out  : r512(plan ws + output bytes) == GPU ws  (PyTorch allocated a second output-sized buffer)
usage: python3 deviceless_match.py <fetch dir>"""
import json, sys, collections
D = sys.argv[1]
KEYS = ("N", "C", "H", "W", "K", "R", "stride", "pad", "dil", "groups")
G = {tuple(r[k] for k in KEYS): r for r in map(json.loads, open(D + "/probe_rtx3090_cudnn9.28.0.42.jsonl")) if "N" in r}
DL = [json.loads(l) for l in open(D + "/inugpu_diag/deviceless_DIAG2_cublas_cudart.jsonl")][1:]
r5 = lambda n: (n + 511) // 512 * 512
eng = lambda n: n.split("_")[0]


def outb(s):
    ho = (s["H"] + 2 * s["pad"] - s["dil"] * (s["R"] - 1) - 1) // s["stride"] + 1
    wo = (s["W"] + 2 * s["pad"] - s["dil"] * (s["R"] - 1) - 1) // s["stride"] + 1
    return 2 * s["N"] * s["K"] * ho * wo


res = collections.Counter(); rows = []; skipped = collections.Counter(); chosen = collections.Counter()
for d in DL:
    g = G[tuple(d[x] for x in KEYS)]; w1 = g["candidates"][0]["ws"]; o = outb(d)
    c = [x for x in d["pt"].get("cands", []) if x.get("built")]
    hit = None
    for i, x in enumerate(c):
        if r5(x["ws"]) == w1:
            hit = (i, "plain"); break
        if r5(x["ws"] + o) == w1:
            hit = (i, "+out"); break
    kind = "1x1" if d["R"] == 1 else ("dw" if d["groups"] == d["C"] and d["groups"] > 1 else ("grouped" if d["groups"] > 1 else f"{d['R']}x{d['R']}"))
    if hit is None:
        res[("none", kind)] += 1
    else:
        res[(hit[1], "pos0" if hit[0] == 0 else "pos>0")] += 1
        chosen[eng(c[hit[0]]["name"])] += 1
        for x in c[:hit[0]]:
            skipped[eng(x["name"])] += 1
    rows.append({**{k: d[k] for k in KEYS}, "gpu_ws": w1, "out": o, "hit": hit,
                 "first_plan": None if not c else [c[0]["name"], c[0]["ws"]]})
print(sorted(res.items(), key=lambda t: -t[1]))
print("chosen engines", chosen.most_common(10))
print("skipped before chosen", skipped.most_common(12))
json.dump(rows, open("deviceless_match_rows.json", "w"))
