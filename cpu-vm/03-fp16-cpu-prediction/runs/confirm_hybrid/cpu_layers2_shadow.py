"""(name ends in shadow.py on purpose) CPU layer labelling v2: exact call boundaries.
After every counted torch-function call a marker request of a unique size is allocated and freed, so the profiler
stream shows where each call ends. An output record (ptr, nbytes) of call c is matched to the LATEST allocation at that
address and size before c's end marker that is still alive at the marker (CPU malloc reuses addresses, so an earlier
same-size temporary at the same address is not the output). Everything else = library layer. Marker events are removed;
the remaining stream must equal the production FP16 stream (stored npz), then labels are attached by alloc order.
usage: cpu_layers2_shadow.py <case_id> <cases.jsonl> <stored case dir> <threads> <out.npz>"""
import gc, json, sys
from pathlib import Path
import numpy as np
import torch
from cgf680 import protocol2 as P
from cgf680.cpu_worker import MARK_BASE, MARKERS, Session, normalize, parse_trace
from cgf680.cpu_worker2 import REGION
from cgf680.shadow import emulate_fp16, match_records, patch_fp16_branches
from cgf680.gemm_shadow import NamedShadow

CALLMARK = 999_983          # unique odd size (bytes)


class MarkShadow(NamedShadow):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        before = self.calls
        res = super().__torch_function__(func, types, args, kwargs)
        if self.calls > before:
            x = torch.empty(CALLMARK, dtype=torch.uint8); del x
        return res


cid, cases_path, stored, threads, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
case = next(c for c in map(json.loads, Path(cases_path).read_text().splitlines()) if c["case_id"] == cid)
models = json.loads(Path("/exp/manifest/models.json").read_text())["models"]
torch.set_num_threads(threads); torch.manual_seed(P.SEED)
m16 = P.load(case, models[case["model"]], torch.float16); del m16; gc.collect()
m = P.finish_load(P.load(case, models[case["model"]], torch.float32), torch.float32); patch_fp16_branches(m)
host, _ = P.host_inputs(case, m.config)
inputs = {k: (v.to(torch.float32) if v.is_floating_point() else v) for k, v in host.items()}
sh = MarkShadow(abort=False)
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
act, addr, size, aid = ev["action"], ev["addr"], ev["size"], ev["alloc_id"]
# call end positions (index of the k-th marker alloc) and removal mask
markpos = [i for i in range(len(act)) if act[i] == 1 and int(size[i]) == CALLMARK]
mark_ids = {int(aid[i]) for i in markpos}
keep = np.array([int(aid[i]) not in mark_ids for i in range(len(act))])
free_at = {}
for i in range(len(act)):
    if act[i] == 0:
        free_at.setdefault(int(aid[i]), i)
by_addr = {}
for i in range(len(act)):
    if act[i] == 1 and keep[i]:
        by_addr.setdefault(int(addr[i]), []).append(i)
label = {}                                # old event index -> 0 model
import bisect
for kind, p, nb, x, call in sh.records:
    if call - 1 >= len(markpos):
        continue
    end = markpos[call - 1]
    lst = by_addr.get(int(p), [])
    j = bisect.bisect_left(lst, end) - 1
    while j >= 0:
        i = lst[j]
        if int(size[i]) == int(nb) and free_at.get(int(aid[i]), 1 << 62) > end:
            label[i] = 0
            break
        j -= 1
# production FP16 sizes on the marker-free stream
ev2 = {k: v[keep] for k, v in ev.items()}
oldidx = np.nonzero(keep)[0]
recs = [r for r in sh.records]
tags, bsrc, _ = match_records(ev2, recs)
ev16, _, idx = emulate_fp16(ev2, tags, bsrc)
mpos2 = {kk: int(np.searchsorted(oldidx, v)) for kk, v in mpos.items()}
p0, p5 = idx[mpos2[0]], idx[mpos2[5]]
lab_by_aid = {int(ev2["alloc_id"][j]): (0 if int(oldidx[j]) in label else 1) for j in range(len(ev2["action"])) if ev2["action"][j] == 1}
a16, s16, id16 = ev16["action"][p0:p5], ev16["size"][p0:p5], ev16["alloc_id"][p0:p5]
layer = np.array([lab_by_aid.get(int(k), 0) for k in id16], np.int8)
z = np.load(Path(stored) / "invocation_fp16_requests.npz")
same = bool(len(a16) == len(z["action"]) and np.array_equal(a16, z["action"]) and np.array_equal(s16, z["size"]))
np.savez_compressed(out, action=a16, size=s16, alloc_id=id16, layer=layer, func=np.zeros(len(a16), np.int32),
                    func_names=np.array(["?"]), load_events=z["load_events"], equal_to_stored=same)
M = 1 << 20
al = a16 == 1
print("CPU_LAYERS2 " + json.dumps({"case_id": cid, "events": int(len(a16)), "equal_to_stored": same, "call_markers": len(markpos),
      "records": len(sh.records), "model_layer_alloc_MiB": round(float(s16[al & (layer == 0)].sum()) / M, 1),
      "library_layer_alloc_MiB": round(float(s16[al & (layer == 1)].sum()) / M, 1)}), flush=True)
