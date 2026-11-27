"""(file name must end in shadow.py: shadow._caller_frame skips such frames)
Shadow FP16 tag check: run the same small case (a) in REAL FP16 on CPU and (b) in FP32 under ShadowMode
(cpu_worker2 path, T5 branch patches). Align torch-function calls in order and list calls whose output dtype in
the real FP16 run disagrees with the shadow tag ('h' = model dtype fp16, 'f' = fp32), with source location.
usage: tagcheck.py <case_id> <cases.jsonl> <batch> <input_len> <new_tokens> <out.json>"""
import json, sys, linecache
from collections import Counter
from pathlib import Path
import torch
from torch.overrides import TorchFunctionMode
from cgf680 import protocol2 as P
from cgf680.shadow import ShadowMode, patch_fp16_branches, shadow_of, _tensors

cid, cases_path, B, L, N, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
case = dict(next(c for c in map(json.loads, Path(cases_path).read_text().splitlines()) if c["case_id"] == cid))
case.update(batch_size=B, input_len=L, new_tokens=N)
pin = json.loads(Path("/exp/manifest/models.json").read_text())["models"][case["model"]]
torch.set_num_threads(8)

def where():
    f = sys._getframe(2)
    while f is not None and ("/torch/" in f.f_code.co_filename or f.f_code.co_filename.endswith("shadow.py")):
        f = f.f_back
    if f is None:
        return "?"
    return f"{Path(f.f_code.co_filename).name}:{f.f_lineno} {linecache.getline(f.f_code.co_filename, f.f_lineno).strip()[:110]}"

def fname(func):
    return getattr(func, "__name__", str(func))

class Real(TorchFunctionMode):
    def __init__(self): super().__init__(); self.log = []
    def __torch_function__(self, func, types, args=(), kwargs=None):
        res = func(*args, **(kwargs or {}))
        outs = [o for o in _tensors(res, []) if o.is_floating_point()]
        self.log.append((fname(func), [(str(o.dtype).replace("torch.", ""), tuple(o.shape)) for o in outs], where()))
        return res

class Shadow(ShadowMode):
    def __init__(self): super().__init__(); self.log = []
    def __torch_function__(self, func, types, args=(), kwargs=None):
        res = super().__torch_function__(func, types, args, kwargs)
        outs = [o for o in _tensors(res, []) if o.is_floating_point()]
        self.log.append((fname(func), [(shadow_of(o), tuple(o.shape)) for o in outs], where()))
        return res

def run(dtype, mode):
    torch.manual_seed(P.SEED)
    m = P.finish_load(P.load(case, pin, dtype), dtype)
    if dtype == torch.float32:
        patch_fp16_branches(m)
    host, _ = P.host_inputs(case, m.config)
    inputs = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in host.items()}
    if case["kind"] != "vision_cls":
        P.generation_config(m, case)
    with mode, torch.inference_mode():
        P.invoke(m, case, inputs)
    return mode.log

real = [x for x in run(torch.float16, Real()) if x[0] != "__get__"]
sh = [x for x in run(torch.float32, Shadow()) if x[0] != "__get__"]
names_r = [x[0] for x in real]; names_s = [x[0] for x in sh]
mis = Counter(); ex = {}
n = min(len(real), len(sh)); aligned = 0
for i in range(n):
    if real[i][0] != sh[i][0]:
        break
    aligned += 1
    for (dt, shp), (tg, shp2) in zip(real[i][1], sh[i][1]):
        want = "h" if dt == "float16" else ("f" if dt == "float32" else dt)
        if want != tg:
            k = (real[i][0], f"real={dt} shadow={tg}", sh[i][2])
            mis[k] += 1
            ex.setdefault(k, shp)
R = {"case": cid, "B": B, "L": L, "N": N, "calls_real": len(real), "calls_shadow": len(sh), "aligned_prefix": aligned,
     "first_divergence": None if aligned == n else [real[aligned][0], real[aligned][2], sh[aligned][0], sh[aligned][2]],
     "mismatches": [[k[0], k[1], k[2], c, ex[k]] for k, c in mis.most_common()]}
Path(out).write_text(json.dumps(R, indent=1))
print(json.dumps({k: R[k] for k in ("calls_real", "calls_shadow", "aligned_prefix", "first_divergence")}))
for r in R["mismatches"][:25]:
    print(r)
