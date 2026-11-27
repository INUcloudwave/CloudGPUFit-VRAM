"""(name ends in shadow.py on purpose) T5 encoder clamp: is its output record matched to the profiler alloc event, and
which FP16-run tag does that event finally get? Small variant of a case, same pipeline as cpu_worker2."""
import gc, json, sys
from pathlib import Path
import torch
from cgf680 import protocol2 as P
from cgf680.cpu_worker import MARK_BASE, MARKERS, Session, normalize, parse_trace
from cgf680.cpu_worker2 import REGION
from cgf680.shadow import match_records, patch_fp16_branches
from cgf680.gemm_shadow import NamedShadow
from cgf680.gemm_probe import record_hits
cid, cases_path, B, L, N = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
case = dict(next(c for c in map(json.loads, Path(cases_path).read_text().splitlines()) if c["case_id"] == cid))
case.update(batch_size=B, input_len=L, new_tokens=N)
pin = json.loads(Path("/exp/manifest/models.json").read_text())["models"][case["model"]]
torch.set_num_threads(4); torch.manual_seed(P.SEED)
m = P.finish_load(P.load(case, pin, torch.float32), torch.float32); patch_fp16_branches(m)
host, _ = P.host_inputs(case, m.config)
P.generation_config(m, case)
sh = NamedShadow(abort=False)
gc.collect(); gc.disable(); sess = Session("user"); sess.start()
try:
    x = torch.empty(MARK_BASE, dtype=torch.uint8); del x
    with torch.profiler.record_function(REGION):
        with sh, torch.inference_mode():
            P.invoke(m, case, host)
finally:
    sess.stop(); gc.enable()
raw = Path("/tmp/t.json"); sess.export(str(raw))
regions, mem = parse_trace(raw); ev, mpos, _ = normalize(regions, mem, len(MARKERS))
tags, buf_src, st = match_records(ev, sh.records)
hits = record_hits(ev, sh.records)
hit_calls = {c for k, c in hits.values()}
hid = B * L * m.config.d_model * 4
rows = []
for kind, p, nb, x, call in sh.records:
    if sh.names.get(call) == "clamp" and nb == hid:
        ev_i = next((i for i, (k2, c2) in hits.items() if c2 == call), None)
        rows.append({"call": call, "record_tag": x, "matched": ev_i is not None,
                     "final_tag_of_event": tags[ev_i] if ev_i is not None else None})
# unmatched alloc events of the hidden size: what tag did they inherit?
un = [tags[i] for i in range(len(ev["action"])) if ev["action"][i] == 1 and int(ev["size"][i]) == hid and i not in hits]
from collections import Counter
print(json.dumps({"hidden_fp32_bytes": hid, "clamp_records": len(rows), "clamp_matched": sum(r["matched"] for r in rows),
                  "clamp_tags": Counter(str(r["final_tag_of_event"]) for r in rows),
                  "unmatched_hidden_size_events": len(un), "their_tags": Counter(un), "match_stats": st}, default=str))
# what happens around the first clamp call: neighbouring calls
c0 = rows[0]["call"] if rows else None
if c0:
    print("calls around first clamp:", [(c, sh.names.get(c)) for c in range(c0 - 6, c0 + 4)])
    near = [(i, int(ev["size"][i]), hits.get(i), tags[i]) for i in range(len(ev["action"])) if ev["action"][i] == 1 and int(ev["size"][i]) == hid][:12]
    print("hidden-size alloc events (idx, size, hit(kind,call), tag):", near)
# --- emulated FP16 sizes before each of the first isinf bool allocs (B*L*d bytes), fresh run vs stored npz
import numpy as np
from cgf680.shadow import emulate_fp16
ev16, _, idx = emulate_fp16(ev, tags, buf_src)
p0 = idx[mpos[0]]
def ctx(a, s, nb):
    out = []
    al = [j for j in range(len(a)) if a[j] == 1]
    for t, j in enumerate(al):
        if int(s[j]) == nb and len(out) < 3:
            out.append([round(int(s[al[t - k]]) / 2**20, 3) for k in (4, 3, 2, 1)])
    return out
boolsz = B * L * m.config.d_model
print("fresh  (4 allocs before isinf):", ctx(ev16["action"][p0:], ev16["size"][p0:], boolsz))
if len(sys.argv) > 6:
    z = np.load(sys.argv[6])
    print("stored (4 allocs before isinf):", ctx(z["action"], z["size"], boolsz))
