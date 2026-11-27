"""Synthesize the GPU load-phase request sequence from the checkpoint header and compare with GPU trace."""
import sys, json, struct, glob, os
sys.path.insert(0, "/diag/src")
from cgf_diag.events import load_snapshot
import torch
DT = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8}
def ckpt_entries(model_dir, st):
    if st:
        f = glob.glob(f"{model_dir}/*.safetensors")[0]
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
        h.pop("__metadata__", None)
        return [(k, v["dtype"], v["shape"]) for k, v in h.items()]   # file (header) order
    sd = torch.load(glob.glob(f"{model_dir}/pytorch_model.bin")[0], map_location="meta", weights_only=True)
    m = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16", torch.int64: "I64"}
    return [(k, m[v.dtype], list(v.shape)) for k, v in sd.items()]
cid, model_dir, st = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
d = f"/handoff/reference/traces_attn_default/{cid}"
t = json.load(open(f"{d}/trace.json")); end = next(x for x in t["history"]["boundaries"] if x["name"] == "model_load_end")
ev = load_snapshot(f"{d}/trace.snapshot.pickle.gz")["device_traces"][0]
g = []; a = f = 0
for e in ev:
    if a >= end["allocation.all.allocated"] and f >= end["allocation.all.freed"]: break
    if e["action"] == "alloc": a += 1; g.append(e["size"])
    elif e["action"] == "free_completed": f += 1; g.append(-e["size"])
ents = ckpt_entries(model_dir, st)
import math
def nb(dt, shape): return DT[dt] * math.prod(shape)
syn = []
for k, dt, sh in ents:
    if dt == "F16":
        syn.append(nb(dt, sh))
    else:
        syn += [nb(dt, sh), 2 * math.prod(sh), -nb(dt, sh)]
gw = g[2:] if g[1] == -g[0] else g
print(json.dumps({"cid": cid, "warmup_bytes": g[0], "sum_fp16_ckpt": sum(2 * math.prod(sh) for _, _, sh in ents),
                  "gpu_n": len(gw), "syn_n": len(syn), "equal": gw == syn,
                  "sorted_equal": sorted(gw) == sorted(syn),
                  "gpu_head": gw[:9], "syn_head": syn[:9], "gpu_tail": gw[-6:], "syn_tail": syn[-6:]}))
