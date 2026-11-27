"""cuDNN version effect on candidate-1 workspace, same RTX 3090, same torch 2.6 (INUGPU 2026-11-23).
usage: python3 version_compare.py <fetch dir>"""
import json, sys, collections
D = sys.argv[1]
KEYS = ("N", "C", "H", "W", "K", "R", "stride", "pad", "dil", "groups")
V = ["9.1.0", "9.5.1.17", "9.7.1.26", "9.10.2.21", "9.12.0.46", "9.28.0.42"]
T = {}
for v in V:
    rows = [json.loads(l) for l in open(f"{D}/probe_rtx3090_cudnn{v}.jsonl")][1:]
    T[v] = {tuple(r[k] for k in KEYS): (r["candidates"][0]["ws"], r["src"]) for r in rows if r["status"] == "ok"}
base = T["9.1.0"]
M = 1 << 20
out = {}
for v in V[1:]:
    same = sum(T[v][k][0] == base[k][0] for k in base)
    d = [T[v][k][0] - base[k][0] for k in base]
    big = sum(abs(x) > 32 * M for x in d)
    zero_flip = sum((T[v][k][0] == 0) != (base[k][0] == 0) for k in base)
    by_src = collections.Counter(base[k][1].split(":")[0] for k in base if T[v][k][0] != base[k][0])
    out[v] = {"same": same, "n": len(base), "diff_gt32MiB": big, "zero_nonzero_flip": zero_flip,
              "max_up_MiB": round(max(d) / M, 1), "max_down_MiB": round(min(d) / M, 1), "diff_by_src": dict(by_src)}
    print(v, out[v])
# consecutive versions
for a, b in zip(V, V[1:]):
    print(f"{a} -> {b}: same {sum(T[a][k][0] == T[b][k][0] for k in base)}/{len(base)}")
json.dump(out, open("version_compare.json", "w"), indent=1)
