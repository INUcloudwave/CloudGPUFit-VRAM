"""Confirmation experiment (CNN request-list cause): run one CNN case on CPU exactly like cpu_worker2 (FP32 + shadow
FP16 sizes, USER_SCOPE profiler), then at the invocation's FP16 allocated peak split live bytes into
  op_output : requests matched to a torch-function output (same on any device)
  internal  : unmatched requests = kernel-internal scratch / reorder buffers of the CPU kernels (mkldnn/oneDNN)
grouped by the torch function that was executing. Also checks the emulated stream equals the stored npz.
usage: python cnn_attr.py <case_id> <cases.jsonl> <npz_root> <threads> <out.jsonl>"""
import gc, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from cgf680 import protocol2 as P
from cgf680.cpu_worker import MARK_BASE, MARKERS, Session, normalize, parse_trace
from cgf680.cpu_worker2 import REGION
from cgf680.shadow import emulate_fp16, match_records, patch_fp16_branches
from cgf680.gemm_shadow import NamedShadow
from cgf680.gemm_probe import record_hits

cid, cases_path, npz_root, threads, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
case = next(c for c in map(json.loads, Path(cases_path).read_text().splitlines()) if c["case_id"] == cid)
models = json.loads(Path("/exp/manifest/models.json").read_text())["models"]
torch.set_num_threads(threads)
torch.manual_seed(P.SEED)
t0 = time.time()
m16 = P.load(case, models[case["model"]], torch.float16); del m16; gc.collect()
m = P.finish_load(P.load(case, models[case["model"]], torch.float32), torch.float32)
patch_fp16_branches(m)
host, _ = P.host_inputs(case, m.config)
inputs = {k: (v.to(torch.float32) if v.is_floating_point() else v) for k, v in host.items()}
shadow = NamedShadow(abort=False)
gc.collect(); gc.disable()
sess = Session("user"); sess.start()
try:
    x = torch.empty(MARK_BASE, dtype=torch.uint8); del x
    with torch.profiler.record_function(REGION):
        with shadow:
            with torch.inference_mode():
                res = P.invoke(m, case, inputs)
    del res
finally:
    sess.stop(); gc.enable()
raw = Path(out).with_suffix(f".{cid}.t{threads}.trace.json")
sess.export(str(raw)); del sess
regions, mem = parse_trace(raw); raw.unlink()
ev, mpos, _ = normalize(regions, mem, len(MARKERS))
tags, buf_src, _ = match_records(ev, shadow.records)
ev16, _, idx = emulate_fp16(ev, tags, buf_src)
hits = record_hits(ev, shadow.records)
# attribution of every old alloc event: matched -> (output, call); unmatched -> (internal, next matched call)
alloc_old = [i for i in range(len(ev["action"])) if ev["action"][i] == 1]
call_of, kind_of = {}, {}
nxt = None
for i in reversed(alloc_old):
    if i in hits:
        nxt = hits[i][1]; kind_of[i] = "op_output"; call_of[i] = hits[i][1]
    else:
        kind_of[i] = "internal"; call_of[i] = nxt
aid_old = {int(ev["alloc_id"][i]): i for i in alloc_old}
p0 = idx[mpos[0]]
a16, s16, id16 = ev16["action"], ev16["size"], ev16["alloc_id"]
z = np.load(Path(npz_root) / cid / "invocation_fp16_requests.npz")
na = len(z["action"])
pa, ps = a16[p0:p0 + na], s16[p0:p0 + na]
same = bool(len(pa) == na and np.array_equal(pa, z["action"]) and np.array_equal(ps, z["size"]))
first_diff = None if same else next((i for i in range(min(len(pa), na)) if pa[i] != z["action"][i] or ps[i] != z["size"][i]), min(len(pa), na))
live, cur, peak, peak_live = {}, 0, -1, None
for j in range(p0, len(a16)):
    k = int(id16[j])
    if a16[j] == 1:
        live[k] = int(s16[j]); cur += int(s16[j])
        if cur > peak:
            peak, peak_live = cur, dict(live)
    elif k in live:
        cur -= live.pop(k)
by = defaultdict(int)
for k, s in peak_live.items():
    i = aid_old.get(k)
    kind = kind_of.get(i, "unknown")
    by[(kind, shadow.names.get(call_of.get(i), "?"))] += s
M = 1 << 20
row = {"case_id": cid, "threads": threads, "stream_equal_to_npz": same, "first_diff": first_diff, "npz_len": na, "probe_len": int(len(a16) - p0), "invocation_peak_MiB": round(peak / M, 1),
       "op_output_MiB": round(sum(v for (k, _), v in by.items() if k == "op_output") / M, 1),
       "internal_MiB": round(sum(v for (k, _), v in by.items() if k == "internal") / M, 1),
       "unknown_MiB": round(sum(v for (k, _), v in by.items() if k == "unknown") / M, 1),
       "top": sorted(([k, n, round(v / M, 1)] for (k, n), v in by.items()), key=lambda x: -x[2])[:10],
       "wall_s": round(time.time() - t0, 1)}
print("CNN_ATTR " + json.dumps(row), flush=True)
open(out, "a").write(json.dumps(row) + "\n")
