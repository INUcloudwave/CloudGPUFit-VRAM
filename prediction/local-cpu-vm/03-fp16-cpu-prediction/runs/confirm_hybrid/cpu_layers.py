"""Hybrid check, CPU side: re-trace one case exactly like cpu_worker2 (FP32 + shadow v1 FP16 sizes) and save the FP16
invocation stream with a per-allocation layer label:
  layer 0 = model layer   (matched to a torch-function output or an emulated dtype-cast buffer: device independent)
  layer 1 = library layer (unmatched = CPU kernel-internal buffer, e.g. mkldnn)
plus the torch function that was executing. usage: cpu_layers.py <case_id> <cases.jsonl> <stored case dir> <threads> <out.npz>"""
import gc, json, sys
from pathlib import Path
import numpy as np
import torch
from cgf680 import protocol2 as P
from cgf680.cpu_worker import MARK_BASE, MARKERS, Session, normalize, parse_trace
from cgf680.cpu_worker2 import REGION
from cgf680.shadow import emulate_fp16, match_records, patch_fp16_branches
from cgf680.gemm_shadow import NamedShadow
from cgf680.gemm_probe import record_hits

cid, cases_path, stored, threads, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
case = next(c for c in map(json.loads, Path(cases_path).read_text().splitlines()) if c["case_id"] == cid)
models = json.loads(Path("/exp/manifest/models.json").read_text())["models"]
torch.set_num_threads(threads); torch.manual_seed(P.SEED)
m16 = P.load(case, models[case["model"]], torch.float16); del m16; gc.collect()
m = P.finish_load(P.load(case, models[case["model"]], torch.float32), torch.float32); patch_fp16_branches(m)
host, _ = P.host_inputs(case, m.config)
inputs = {k: (v.to(torch.float32) if v.is_floating_point() else v) for k, v in host.items()}
sh = NamedShadow(abort=False)
gc.collect(); gc.disable(); sess = Session("user"); sess.start()
try:
    x = torch.empty(MARK_BASE, dtype=torch.uint8); del x
    with torch.profiler.record_function(REGION):
        with sh, torch.inference_mode():
            gi, o, new = P.invoke(m, case, inputs)
    x = torch.empty(MARK_BASE + 8, dtype=torch.uint8); del x
    del gi, o
    x = torch.empty(MARK_BASE + 10, dtype=torch.uint8); del x
finally:
    sess.stop(); gc.enable()
raw = Path(f"/tmp/{cid}.json"); sess.export(str(raw))
regions, mem = parse_trace(raw); raw.unlink()
ev, mpos, _ = normalize(regions, mem, len(MARKERS))
tags, bsrc, _ = match_records(ev, sh.records)
ev16, _, idx = emulate_fp16(ev, tags, bsrc)
hits = record_hits(ev, sh.records)
# label per OLD alloc event, then carry to the FP16 stream through alloc_id
lab, fn = {}, {}
nxt = None
for i in range(len(ev["action"]) - 1, -1, -1):
    if ev["action"][i] != 1:
        continue
    aid = int(ev["alloc_id"][i])
    if i in hits:
        lab[aid] = 0; nxt = sh.names.get(hits[i][1], "?") if hits[i][0] != "buf" else (nxt or "cast")
        fn[aid] = sh.names.get(hits[i][1], "cast") if hits[i][0] != "buf" else "cast"
    else:
        lab[aid] = 1; fn[aid] = nxt or "?"
p0, p5 = idx[mpos[0]], idx[mpos[5]]
a16, s16, id16 = ev16["action"][p0:p5], ev16["size"][p0:p5], ev16["alloc_id"][p0:p5]
layer = np.array([lab.get(int(k), 0) for k in id16], np.int8)
names = sorted(set(fn.values())); nidx = {n: i for i, n in enumerate(names)}
func = np.array([nidx.get(fn.get(int(k), "?"), -1) for k in id16], np.int32)
z = np.load(Path(stored) / "invocation_fp16_requests.npz")
same = bool(np.array_equal(a16, z["action"]) and np.array_equal(s16, z["size"]))
np.savez_compressed(out, action=a16, size=s16, alloc_id=id16, layer=layer, func=func, func_names=np.array(names),
                    load_events=z["load_events"], equal_to_stored=same)
M = 1 << 20
al = a16 == 1
print("CPU_LAYERS " + json.dumps({"case_id": cid, "events": int(len(a16)), "equal_to_stored": same,
      "model_layer_alloc_MiB": round(float(s16[al & (layer == 0)].sum()) / M, 1),
      "library_layer_alloc_MiB": round(float(s16[al & (layer == 1)].sum()) / M, 1),
      "library_layer_allocs": int((al & (layer == 1)).sum())}), flush=True)
