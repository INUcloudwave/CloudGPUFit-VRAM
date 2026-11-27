"""Predicted reserved (RTX 3090 capacity) of the 108 CNN conditions with each cuDNN version's candidate-1 table, vs 9.1.0.
usage (image): python version_impact.py <fetch dir mounted> """
import json, sys
from multiprocessing import Pool
sys.argv = [sys.argv[0], "/tmp/unused"] + sys.argv[1:]
sys.path.insert(0, "/exp/results/reserved_v3/fallback")
import fallback_eval as fe
from cgf680 import replay3
D = sys.argv[2]
V = ["9.1.0", "9.5.1.17", "9.10.2.21", "9.28.0.42"]
TAB = {}
for v in V:
    t = {}
    for line in open(f"{D}/probe_rtx3090_cudnn{v}.jsonl"):
        r = json.loads(line)
        if r.get("status") == "ok":
            t[tuple(r[k] for k in fe.KEYS)] = [c["ws"] for c in r["candidates"] if not c.get("oom")]
    TAB[v] = t


def one(c):
    out = {"case_id": c}
    for v in V:
        fe.CANDS.clear(); fe.CANDS.update(TAB[v])
        s3, w3, *_ = fe.streams(fe.lab_of(c), fe.pos.get(c))
        r = replay3.replay(s3, fe.CAP, "descending", w3, fe.WSB)
        out[v] = None if r["oom"] else r["m0_start_reserved"]
    return out


cases = sorted({p.stem for p in (fe.E / "runs/layer_labels").glob("*.npz")} | {p.stem for p in (fe.E / "runs/layer_labels_newmodels").glob("*.npz")})
cases = [c for c in cases if not c.startswith("vit")]
with Pool(8) as p:
    rows = p.map(one, cases)
M = 1 << 20
res = {}
for v in V[1:]:
    d = [(r[v] - r["9.1.0"], r["case_id"]) for r in rows]
    res[v] = {"same": sum(x == 0 for x, _ in d), "within100MB": sum(abs(x) <= 1e8 for x, _ in d), "n": len(d),
              "max_up_MiB": max(d)[0] // M, "max_up_case": max(d)[1], "max_down_MiB": min(d)[0] // M, "max_down_case": min(d)[1]}
    print(v, res[v], flush=True)
json.dump({"rows": rows, "summary": res}, open("/exp/results/reserved_v3/cudnn_version/version_impact.json", "w"), indent=1)
