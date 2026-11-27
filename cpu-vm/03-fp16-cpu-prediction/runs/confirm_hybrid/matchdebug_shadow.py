"""Why are some ResNet op outputs unmatched? Trace a small variant, then for every unmatched 'out' record list the alloc
events at the same address (index, size, free index) and the matcher floor at that moment."""
import gc, json, sys, bisect
from pathlib import Path
import torch
from cgf680 import protocol2 as P
from cgf680.cpu_worker import MARK_BASE, MARKERS, Session, normalize, parse_trace
from cgf680.cpu_worker2 import REGION
from cgf680.shadow import patch_fp16_branches
from cgf680.gemm_shadow import NamedShadow
cid, cases_path, B, R, threads = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
case = dict(next(c for c in map(json.loads, Path(cases_path).read_text().splitlines()) if c["case_id"] == cid))
case.update(batch_size=B, resolution=R)
pin = json.loads(Path("/exp/manifest/models.json").read_text())["models"][case["model"]]
torch.set_num_threads(threads); torch.manual_seed(P.SEED)
m = P.finish_load(P.load(case, pin, torch.float32), torch.float32); patch_fp16_branches(m)
host, _ = P.host_inputs(case, m.config)
sh = NamedShadow(abort=False)
gc.collect(); gc.disable(); sess = Session("user"); sess.start()
try:
    x = torch.empty(MARK_BASE, dtype=torch.uint8); del x
    with torch.profiler.record_function(REGION):
        with sh, torch.inference_mode():
            P.invoke(m, case, {k: (v.to(torch.float32) if v.is_floating_point() else v) for k, v in host.items()})
finally:
    sess.stop(); gc.enable()
raw = Path("/tmp/t.json"); sess.export(str(raw))
regions, mem = parse_trace(raw); ev, mpos, _ = normalize(regions, mem, len(MARKERS))
act, addr, size, aid = ev["action"], ev["addr"], ev["size"], ev["alloc_id"]
free_of = {}
for i in range(len(act)):
    if act[i] == 0:
        free_of.setdefault(int(aid[i]), i)
by_addr = {}
for i in range(len(act)):
    if act[i] == 1:
        by_addr.setdefault(int(addr[i]), []).append(i)
used, floor, cur, cmax = set(), -1, None, -1
un = []
for kind, p, nb, x, call in sh.records:
    if call != cur:
        floor = max(floor, cmax); cur = call
    lst = by_addr.get(int(p), []); j = bisect.bisect_right(lst, floor); hit = None
    while j < len(lst):
        k = lst[j]
        if k not in used and int(size[k]) == int(nb):
            hit = k; break
        j += 1
    if hit is None:
        cands = [(k, int(size[k]), free_of.get(int(aid[k])), k in used) for k in lst]
        un.append({"call": call, "func": sh.names.get(call), "kind": kind, "nbytes": int(nb), "floor": floor, "cands_at_addr": cands[:8]})
        continue
    used.add(hit); cmax = max(cmax, hit)
big = [u for u in un if u["nbytes"] > 100000]
print(json.dumps({"records": len(sh.records), "unmatched": len(un), "unmatched_big": len(big)}))
for u in big[:12]:
    print(json.dumps(u))
