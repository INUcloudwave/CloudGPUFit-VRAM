"""CPU request layer labels for a stored CPU case (model layer vs CPU library layer), with the torch function of the
enclosing call. Generalised from results/reserved_v3/confirm/hybrid/cpu_layers2_shadow.py (2026-11-17, verified:
CPU model layer == GPU model layer on 3/3 GPU traces). The file name ends in "shadow.py" on purpose:
shadow._caller_frame skips frames of such files when it reads the calling source line.

Method (same as cpu_layers2): after every counted torch-function call a marker request of a unique size (CALLMARK) is
allocated and freed, so the profiler stream shows where each call ends. An output record (ptr, nbytes) of call c is
matched to the LATEST allocation at that address and size before c's end marker that is still alive at the marker
(CPU malloc reuses addresses). Everything else is the library layer (CPU kernel scratch: oneDNN/mkldnn buffers, ATen
internal temporaries). Marker events are removed; the remaining FP16-sized stream must equal the stored production
stream (invocation_fp16_requests.npz), then labels are attached by allocation id.

Added here: every allocation also gets the name of the torch function call during which it was allocated (`func`
index into `func_names`), so a predictor can drop only library requests made inside selected calls (e.g. conv2d),
and the conv parameters of every conv call (for workspace models).

usage (inside the pinned image):
  python -m cgf680.layer_shadow <case_id> <cases.jsonl> <stored case dir> <threads> <out.npz>
"""
from __future__ import annotations

import bisect
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

from cgf680 import protocol2 as P
from cgf680.cpu_worker import MARK_BASE, MARKERS, Session, normalize, parse_trace
from cgf680.cpu_worker2 import REGION
from cgf680.gemm_shadow import NamedShadow
from cgf680.shadow import emulate_fp16, match_records, patch_fp16_branches

CALLMARK = 999_983          # unique odd size (bytes), never requested by a model
CONV_NAMES = {"conv2d", "conv1d", "conv3d", "_convolution", "convolution", "Conv2d.forward"}


class MarkShadow(NamedShadow):
    """NamedShadow + call-end marker + conv parameters per call."""

    def __init__(self):
        super().__init__(abort=False)
        self.conv = {}

    def __torch_function__(self, func, types, args=(), kwargs=None):
        before = self.calls
        res = super().__torch_function__(func, types, args, kwargs)
        if self.calls > before:
            n = self.names.get(self.calls, "")
            if n in CONV_NAMES and len(args) >= 2 and hasattr(args[0], "shape"):
                kw = dict(kwargs or {})
                rest = list(args[2:]) + [None] * 5
                keys = ("bias", "stride", "padding", "dilation", "groups")
                p = {k: kw.get(k, rest[i]) for i, k in enumerate(keys)}
                self.conv[self.calls] = {
                    "input": list(args[0].shape), "weight": list(args[1].shape),
                    "stride": _tup(p["stride"], 1), "padding": _tup(p["padding"], 0),
                    "dilation": _tup(p["dilation"], 1), "groups": int(p["groups"] or 1),
                    "output": list(res.shape) if hasattr(res, "shape") else None}
            x = torch.empty(CALLMARK, dtype=torch.uint8)
            del x
        return res


def _tup(v, d):
    if v is None:
        return [d, d]
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return [v, v]
    return [int(t) for t in v]


def label_case(case, models, stored, threads):
    torch.set_num_threads(threads)
    torch.manual_seed(P.SEED)
    m16 = P.load(case, models[case["model"]], torch.float16)
    del m16
    gc.collect()
    m = P.finish_load(P.load(case, models[case["model"]], torch.float32), torch.float32)
    patch_fp16_branches(m)
    host, _ = P.host_inputs(case, m.config)
    inputs = {k: (v.to(torch.float32) if v.is_floating_point() else v) for k, v in host.items()}
    sh = MarkShadow()
    gc.collect()
    gc.disable()
    sess = Session("user")
    sess.start()
    try:
        x = torch.empty(MARK_BASE, dtype=torch.uint8)
        del x
        with torch.profiler.record_function(REGION):
            with sh, torch.inference_mode():
                gi, o, new = P.invoke(m, case, inputs)
        x = torch.empty(MARK_BASE + 8, dtype=torch.uint8)
        del x
        del gi, o
        x = torch.empty(MARK_BASE + 10, dtype=torch.uint8)
        del x
    finally:
        sess.stop()
        gc.enable()
    raw = Path(f"/tmp/{case['case_id']}.json")
    sess.export(str(raw))
    regions, mem = parse_trace(raw)
    raw.unlink()
    ev, mpos, _ = normalize(regions, mem, len(MARKERS))
    act, addr, size, aid = ev["action"], ev["addr"], ev["size"], ev["alloc_id"]
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
    model_layer = set()
    for kind, p, nb, xx, call in sh.records:
        if call - 1 >= len(markpos):
            continue
        end = markpos[call - 1]
        lst = by_addr.get(int(p), [])
        j = bisect.bisect_left(lst, end) - 1
        while j >= 0:
            i = lst[j]
            if int(size[i]) == int(nb) and free_at.get(int(aid[i]), 1 << 62) > end:
                model_layer.add(i)
                break
            j -= 1
    # enclosing call of every event: call k+1 runs between marker k and marker k+1
    call_of = np.searchsorted(np.array(markpos, dtype=np.int64), np.arange(len(act)), side="right") + 1
    ev2 = {k: v[keep] for k, v in ev.items()}
    oldidx = np.nonzero(keep)[0]
    tags, bsrc, _ = match_records(ev2, list(sh.records))
    ev16, _, idx = emulate_fp16(ev2, tags, bsrc)
    mpos2 = {kk: int(np.searchsorted(oldidx, v)) for kk, v in mpos.items()}
    p0, p5 = idx[mpos2[0]], idx[mpos2[5]]
    names = sorted(set(sh.names.values()) | {"?"})
    nidx = {n: i for i, n in enumerate(names)}
    lab, fn, cl = {}, {}, {}
    for j in range(len(ev2["action"])):
        if ev2["action"][j] == 1:
            o = int(oldidx[j])
            k = int(ev2["alloc_id"][j])
            lab[k] = 0 if o in model_layer else 1
            c = int(call_of[o])
            cl[k] = c
            fn[k] = nidx.get(sh.names.get(c, "?"), nidx["?"])
    a16, s16, id16 = ev16["action"][p0:p5], ev16["size"][p0:p5], ev16["alloc_id"][p0:p5]
    layer = np.array([lab.get(int(k), 0) for k in id16], np.int8)
    func = np.array([fn.get(int(k), nidx["?"]) for k in id16], np.int32)
    call = np.array([cl.get(int(k), 0) for k in id16], np.int32)
    z = np.load(Path(stored) / "invocation_fp16_requests.npz")
    same = bool(len(a16) == len(z["action"]) and np.array_equal(a16, z["action"]) and np.array_equal(s16, z["size"]))
    return dict(action=a16, size=s16, alloc_id=id16, layer=layer, func=func, call=call, func_names=np.array(names),
                load_events=z["load_events"], equal_to_stored=same,
                conv_json=np.array(json.dumps({str(k): v for k, v in sh.conv.items()})))


def main():
    cid, cases_path, stored, threads, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    case = next(c for c in map(json.loads, Path(cases_path).read_text().splitlines()) if c["case_id"] == cid)
    # optional 6th argument (2026-11-23, generalisation-test models): models manifest; default unchanged
    mpath = sys.argv[6] if len(sys.argv) > 6 else "/exp/manifest/models.json"
    models = json.loads(Path(mpath).read_text())["models"]
    r = label_case(case, models, stored, threads)
    np.savez_compressed(out, **r)
    M = 1 << 20
    al = r["action"] == 1
    names = list(r["func_names"])
    lib = al & (r["layer"] == 1)
    by_func = {}
    for i in np.nonzero(lib)[0]:
        n = names[r["func"][i]]
        by_func[n] = by_func.get(n, 0) + int(r["size"][i])
    print("LAYER_LABELS " + json.dumps({
        "case_id": cid, "events": int(len(r["action"])), "equal_to_stored": r["equal_to_stored"],
        "model_layer_alloc_MiB": round(float(r["size"][al & (r["layer"] == 0)].sum()) / M, 1),
        "library_layer_alloc_MiB": round(float(r["size"][lib].sum()) / M, 1),
        "library_MiB_by_func": {k: round(v / M, 1) for k, v in sorted(by_func.items(), key=lambda t: -t[1])[:8]},
        "conv_calls": len(json.loads(str(r["conv_json"])))}), flush=True)


if __name__ == "__main__":
    main()
