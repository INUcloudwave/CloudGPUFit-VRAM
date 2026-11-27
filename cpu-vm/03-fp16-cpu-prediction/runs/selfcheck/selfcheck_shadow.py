"""(name ends in shadow.py on purpose) FP16 self-check without a GPU.
For one small variant of a case, trace the invocation request stream three ways on CPU:
  real : model and inputs in FP16, no emulation (true FP16 request sizes of CPU kernels)
  v1   : FP32 run + shadow rule v1 (production)          -> emulated FP16 sizes
  v2   : FP32 run + shadow rule v2 (cgf680.rule2_shadow) -> emulated FP16 sizes
and align v1/v2 against real (difflib on the alloc-size sequence). Reports mismatching requests/bytes and the most
frequent (real size -> emulated size) substitutions. CPU FP16 kernels may have their own internal temporaries, so
'real' is the reference for dtype-driven sizes, not for kernel internals.
usage: selfcheck_shadow.py <case_id> <cases.jsonl> <B> <L or R> <N> <threads> <out.jsonl>"""
import difflib, gc, json, sys, time
from collections import Counter
from pathlib import Path
import torch
from cgf680 import protocol2 as P
from cgf680.cpu_worker import MARK_BASE, MARKERS, Session, normalize, parse_trace
from cgf680.cpu_worker2 import REGION
from cgf680.shadow import ShadowMode, emulate_fp16, match_records, patch_fp16_branches
from cgf680.rule2_shadow import ShadowModeV2, match_records_v2

cid, cases_path, B, L, N, threads, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
case = dict(next(c for c in map(json.loads, Path(cases_path).read_text().splitlines()) if c["case_id"] == cid))
case["batch_size"] = B
if case["kind"] == "vision_cls":
    case["resolution"] = L
else:
    case.update(input_len=L, new_tokens=N)
pin = json.loads(Path("/exp/manifest/models.json").read_text())["models"][case["model"]]
torch.set_num_threads(threads)

PATCHED = []

def trace(dtype, shadow):
    torch.manual_seed(P.SEED)
    m = P.finish_load(P.load(case, pin, dtype), dtype)
    if shadow is not None and not PATCHED:
        patch_fp16_branches(m); PATCHED.append(1)   # class-level patch: once per process (after the real FP16 run)
    host, _ = P.host_inputs(case, m.config)
    inputs = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in host.items()}
    if case["kind"] != "vision_cls":
        P.generation_config(m, case)
    gc.collect(); gc.disable(); sess = Session("user"); sess.start()
    try:
        x = torch.empty(MARK_BASE, dtype=torch.uint8); del x
        with torch.profiler.record_function(REGION):
            if shadow is not None:
                with shadow, torch.inference_mode():
                    P.invoke(m, case, inputs)
            else:
                with torch.inference_mode():
                    P.invoke(m, case, inputs)
    finally:
        sess.stop(); gc.enable()
    raw = Path(f"/tmp/{cid}.{dtype}.json"); sess.export(str(raw)); del sess
    regions, mem = parse_trace(raw); raw.unlink()
    ev, mpos, _ = normalize(regions, mem, len(MARKERS))
    del m; gc.collect()
    return ev, mpos

t0 = time.time()
row = {"case_id": cid, "B": B, "L_or_R": L, "N": N, "threads": threads}
ev, mpos = trace(torch.float16, None)
p0 = mpos[0]
real = [int(s) for a, s in zip(ev["action"][p0:], ev["size"][p0:]) if a == 1]
row["real_wall_s"] = round(time.time() - t0, 1)
seqs = {}
for name, mode in (("v1", ShadowMode()), ("v2", ShadowModeV2())):
    ev, mpos = trace(torch.float32, mode)
    if name == "v1":
        tags, bsrc, st = match_records(ev, mode.records)
    else:
        tags, bsrc, st = match_records_v2(ev, mode.records, mode.call_in_tag)
        row["v2_retagged"] = st.get("rule_v2_retagged"); row["v2_retagged_bytes"] = st.get("rule_v2_retagged_bytes")
    ev16, _, idx = emulate_fp16(ev, tags, bsrc)
    q0 = idx[mpos[0]]
    seqs[name] = [int(s) for a, s in zip(ev16["action"][q0:], ev16["size"][q0:]) if a == 1]
for name, seq in seqs.items():
    sm = difflib.SequenceMatcher(None, real, seq, autojunk=False)
    mis_n = mis_b = 0; subs = Counter()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        mis_n += max(i2 - i1, j2 - j1)
        mis_b += abs(sum(real[i1:i2]) - sum(seq[j1:j2])) if tag != "replace" else sum(abs(a - b) for a, b in zip(real[i1:i2], seq[j1:j2])) + abs(sum(real[i1 + (j2 - j1):i2]) - sum(seq[j1 + (i2 - i1):j2]))
        if tag == "replace" and (i2 - i1) == (j2 - j1):
            for a, b in zip(real[i1:i2], seq[j1:j2]):
                subs[(a, b)] += 1
        else:
            subs[(tuple(real[i1:i2][:3]), tuple(seq[j1:j2][:3]))] += 1
    row[name] = {"requests_real": len(real), "requests_emulated": len(seq), "identical": real == seq,
                 "mismatch_requests": mis_n, "mismatch_bytes": int(mis_b),
                 "top_substitutions_real_to_emulated": [[str(k), c] for k, c in subs.most_common(8)]}
row["wall_s"] = round(time.time() - t0, 1)
print("SELFCHECK " + json.dumps(row), flush=True)
open(out, "a").write(json.dumps(row) + "\n")
