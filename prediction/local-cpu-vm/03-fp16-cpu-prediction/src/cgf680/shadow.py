"""FP16 shadow-dtype emulation for an FP32 trace (TorchFunctionMode).

Goal: from an FP32 CPU run, obtain the request trace an FP16 run would produce, by
  (1) tagging every new output storage with the dtype it WOULD have in the FP16 run
      ('h' = follows model dtype -> half size, 'f' = fixed fp32 -> same size, 'x' = non-float/exact),
  (2) reproducing casts that are no-ops in FP32 but real copies in FP16 (x.to(float32) on a model-dtype
      tensor = upcast copy; x.to(float32)/type_as on a fixed-fp32 tensor = downcast copy to model dtype):
      a uint8 buffer of the FP16-run copy size is allocated and attached to the returned alias, so its
      allocation/free appear in the profiler stream at the right place with the right lifetime.
      For downcasts the fp32 source is recorded so the replay frees it when the copy is made.

Records (in call order): ("out", data_ptr, nbytes, tag) for new output storages, ("buf", data_ptr,
nbytes, src_ptr_or_0) for emulated cast buffers. They are matched to profiler [memory] alloc events by
address+size with a monotone cursor (cgf680.shadow_match).
Known gaps (reported, not emulated): branches on `dtype == torch.float16` (T5 inf clamp), CPU kernel
internal temporaries (tagged by the enclosing call), 0-dim type-promotion corner cases.
"""
from __future__ import annotations

import torch
from torch.overrides import TorchFunctionMode

F32 = torch.float32
_TO_FUNCS = {torch.Tensor.to, torch.Tensor.float, torch.Tensor.type_as, torch.Tensor.type}
_SOFTMAX = {torch.nn.functional.softmax, torch.nn.functional.log_softmax, torch.softmax, torch.log_softmax,
            torch.Tensor.softmax, torch.Tensor.log_softmax}
_SKIP = {torch.Tensor.__repr__, torch.Tensor.__format__, torch.Tensor.size, torch.Tensor.dim, torch.Tensor.numel,
         torch.Tensor.__len__, torch.Tensor.data_ptr, torch.Tensor.untyped_storage, torch.Tensor.element_size,
         torch.Tensor.is_floating_point, torch.Tensor.__bool__, torch.Tensor.item, torch.Tensor.tolist,
         torch.Tensor.numpy, torch.Tensor.stride, torch.Tensor.is_contiguous}


def shadow_of(t):
    s = getattr(t, "_cgf_sd", None)
    if s is not None:
        return s
    return "h" if t.is_floating_point() else "x"


def _tensors(obj, out):
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            _tensors(o, out)
    elif isinstance(obj, dict):
        for o in obj.values():
            _tensors(o, out)
    elif hasattr(obj, "to_tuple") and callable(obj.to_tuple):  # ModelOutput
        try:
            _tensors(obj.to_tuple(), out)
        except Exception:
            pass
    return out


def _target_dtype(func, args, kwargs):
    if func is torch.Tensor.float:
        return F32
    if func is torch.Tensor.type_as:
        return args[1].dtype
    d = kwargs.get("dtype")
    if d is None:
        for a in args[1:]:
            if isinstance(a, torch.dtype):
                d = a
            elif isinstance(a, torch.Tensor) and func is torch.Tensor.to:
                d = a.dtype
    return d


_POS_CACHE = {}
_EXPLICIT_CACHE = {}
_FP32_RE = None


def _caller_frame():
    import sys
    f = sys._getframe(2)
    while f is not None and ("/torch/" in f.f_code.co_filename or f.f_code.co_filename.endswith("shadow.py")):
        f = f.f_back
    return f


def explicit_fp32_at_call(method_names=(".to(", ".type(")):
    """True if the source text of the calling expression names float32 literally (torch.float32,
    torch.float, .float()) in the argument list of the dtype-taking call; False if the dtype comes from
    a variable/attribute (e.g. query.dtype, input_dtype, self.weight.dtype) = model dtype in FP16 run.
    Uses Python 3.11 co_positions of the current instruction."""
    import linecache
    import re
    global _FP32_RE
    if _FP32_RE is None:
        _FP32_RE = re.compile(r"float32|torch\.float\b(?!16)")
    f = _caller_frame()
    if f is None:
        return None
    key = (f.f_code, f.f_lasti)
    if key in _EXPLICIT_CACHE:
        return _EXPLICIT_CACHE[key]
    pos = _POS_CACHE.get(f.f_code)
    if pos is None:
        pos = _POS_CACHE[f.f_code] = list(f.f_code.co_positions())
    try:
        ln, eln, col, ecol = pos[f.f_lasti // 2]
        lines = [linecache.getline(f.f_code.co_filename, i) for i in range(ln, eln + 1)]
        if len(lines) == 1:
            seg = lines[0][col:ecol]
        else:
            seg = lines[0][col:] + "".join(lines[1:-1]) + lines[-1][:ecol]
    except Exception:
        _EXPLICIT_CACHE[key] = None
        return None
    cut = max(seg.rfind(m) for m in method_names)
    argtext = seg[cut:] if cut >= 0 else seg
    if "dtype=" in seg and cut < 0:
        argtext = seg[seg.rfind("dtype="):]
    val = bool(_FP32_RE.search(argtext))
    _EXPLICIT_CACHE[key] = val
    return val


class ShadowMode(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.records = []
        self.calls = 0
        self.emulated = {"upcast": 0, "downcast": 0, "upcast_bytes": 0, "downcast_bytes": 0}
        self.undetermined_casts = 0
        self.enabled = True

    def _emulate(self, src, res, want, in_ptrs):
        s = shadow_of(src)
        same = res.untyped_storage().data_ptr() == src.untyped_storage().data_ptr()
        if not same:  # a real new tensor in the FP32 run too (copy=True, int->float, ...)
            res._cgf_sd = want
            self._record_outputs([res], in_ptrs)
            return res
        if want == s:  # no-op in both runs
            return res
        alias = res.view_as(res) if res is src else res
        nbytes = src.numel() * (4 if want == "f" else 2)
        buf = torch.empty(nbytes, dtype=torch.uint8)
        alias._cgf_buf = buf
        alias._cgf_sd = want
        kind = "upcast" if want == "f" else "downcast"
        self.emulated[kind] += 1
        self.emulated[kind + "_bytes"] += nbytes
        self.records.append(("buf", buf.data_ptr(), nbytes,
                             src.untyped_storage().data_ptr() if kind == "downcast" else 0, self.calls))
        return alias

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if not self.enabled or func in _SKIP:
            return func(*args, **kwargs)
        ins = _tensors(list(args) + list(kwargs.values()), [])
        in_ptrs = {t.untyped_storage().data_ptr() for t in ins}
        res = func(*args, **kwargs)
        self.calls += 1
        if func in _TO_FUNCS and ins and isinstance(res, torch.Tensor) and isinstance(args[0], torch.Tensor):
            src = args[0]
            tgt = _target_dtype(func, args, kwargs)
            if res.dtype == F32 and (tgt == F32 or func is torch.Tensor.float):
                if func is torch.Tensor.float:
                    want = "f"
                elif func is torch.Tensor.type_as:
                    want = shadow_of(args[1])
                elif func is torch.Tensor.to and any(isinstance(x, torch.Tensor) for x in args[1:]):
                    want = shadow_of(next(x for x in args[1:] if isinstance(x, torch.Tensor)))
                else:
                    ex = explicit_fp32_at_call()
                    if ex is None:
                        self.undetermined_casts += 1
                        ex = False
                    want = "f" if ex else "h"
                return self._emulate(src, res, want, in_ptrs)
        outs = _tensors(res, [])
        if not outs:
            return res
        if kwargs.get("dtype") == F32:
            ex = explicit_fp32_at_call(method_names=("dtype=",))
            tag = "f" if ex else "h"
        else:
            fl = [shadow_of(t) for t in ins if t.is_floating_point()]
            tag = "f" if "f" in fl else "h"
        for o in outs:
            if o.is_floating_point() and getattr(o, "_cgf_sd", None) is None:
                try:
                    o._cgf_sd = tag
                except Exception:
                    pass
        self._record_outputs(outs, in_ptrs)
        return res

    def _record_outputs(self, outs, in_ptrs):
        for o in outs:
            st = o.untyped_storage()
            p = st.data_ptr()
            if p == 0 or p in in_ptrs:
                continue
            self.records.append(("out", p, st.nbytes(), shadow_of(o) if o.is_floating_point() else "x", self.calls))


# ----------------------------------------------------------------------------- model-specific FP16 branches
def patch_fp16_branches(model):
    """Re-create branches that only execute when the model dtype is float16 (T5): LayerNorm downcast
    and the inf clamp in T5Block. Source is taken from the installed transformers and edited by exact
    string replacement; returns a report (fails loudly if the expected text is absent)."""
    import inspect
    import textwrap
    rep = {}
    mt = getattr(model.config, "model_type", "")
    if mt != "t5":
        return rep
    import transformers.models.t5.modeling_t5 as m

    def em_fp16(t):
        return shadow_of(t) == "h"

    def rebuild(cls, name, repl):
        src = textwrap.dedent(inspect.getsource(getattr(cls, name)))
        for a, b in repl:
            if a not in src:
                raise RuntimeError(f"T5 patch: text not found in {cls.__name__}.{name}: {a}")
            src = src.replace(a, b)
        ns = dict(m.__dict__)
        ns["_cgf_em_fp16"] = em_fp16
        import linecache
        fname = f"<cgf680 patched {cls.__name__}.{name}>"
        linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
        exec(compile(src, fname, "exec"), ns)
        setattr(cls, name, ns[name])
        return src.count("_cgf_em_fp16") + src.count("if True")

    rep["T5LayerNorm.forward"] = rebuild(m.T5LayerNorm, "forward",
                                         [("if self.weight.dtype in [torch.float16, torch.bfloat16]:", "if True:")])
    rep["T5Block.forward"] = rebuild(m.T5Block, "forward",
                                     [("if hidden_states.dtype == torch.float16:", "if _cgf_em_fp16(hidden_states):")])
    return rep


# ----------------------------------------------------------------------------- matching + transform
def match_records(ev, records):
    """Assign an FP16-run tag to every traced request.

    ev: normalized arrays incl. 'addr' (request events in Ev Idx order). Each record is matched to the
    first not-yet-matched alloc event at the same address and size after the highest event index matched
    by EARLIER calls (outputs of one call may be allocated in any order). Unmatched requests take the tag
    of the next matched request (the call that was executing). Returns (tags list, buf_src dict, stats)."""
    import bisect
    import numpy as np
    act, addr, size = ev["action"], ev["addr"], ev["size"]
    alloc_idx = np.nonzero(act == 1)[0]
    by_addr = {}
    for i in alloc_idx.tolist():
        by_addr.setdefault(int(addr[i]), []).append(i)
    used = set()
    tag = {}
    buf_src = {}
    floor, cur_call, call_max = -1, None, -1
    st = {"records": len(records), "matched": 0, "unmatched_records": 0, "buf_records": 0}
    for rec in records:
        kind, p, nbytes, x, call = rec
        if call != cur_call:
            floor = max(floor, call_max)
            cur_call = call
        lst = by_addr.get(int(p), [])
        j = bisect.bisect_right(lst, floor)
        hit = None
        while j < len(lst):
            k = lst[j]
            if k not in used and int(size[k]) == int(nbytes):
                hit = k
                break
            j += 1
        if hit is None:
            st["unmatched_records"] += 1
            continue
        used.add(hit)
        st["matched"] += 1
        call_max = max(call_max, hit)
        if kind == "buf":
            tag[hit] = "b"
            st["buf_records"] += 1
            if x:
                buf_src[hit] = int(x)
        else:
            tag[hit] = x
    tags = [None] * len(act)
    nxt = None
    for i in range(len(act) - 1, -1, -1):
        if act[i] != 1:
            continue
        if i in tag:
            tags[i] = tag[i]
            nxt = tag[i] if tag[i] != "b" else nxt
        else:
            tags[i] = ("inferred_" + nxt) if nxt in ("h", "f", "x") else "inferred_h"
    st["requests"] = int((act == 1).sum())
    st["tag_counts"] = {}
    st["tag_bytes"] = {}
    for i in alloc_idx.tolist():
        t = tags[i]
        st["tag_counts"][t] = st["tag_counts"].get(t, 0) + 1
        st["tag_bytes"][t] = st["tag_bytes"].get(t, 0) + int(size[i])
    return tags, buf_src, st


def emulate_fp16(ev, tags, buf_src):
    """Return new event arrays with FP16-run sizes and early frees of downcast sources."""
    import numpy as np
    act, aid, size, region, addr = ev["action"], ev["alloc_id"], ev["size"], ev["region"], ev["addr"]
    out_a, out_s, out_id, out_r = [], [], [], []
    live_addr = {}   # addr -> alloc_id of live request
    early = set()
    newsize = {}
    new_index_of_old = []
    for i in range(len(act)):
        new_index_of_old.append(len(out_a))
        if act[i] == 1:
            t = tags[i] or "inferred_h"
            s = int(size[i])
            if t.endswith("h"):
                s = (s + 1) // 2
            newsize[int(aid[i])] = s
            live_addr[int(addr[i])] = int(aid[i])
            out_a.append(1); out_s.append(s); out_id.append(int(aid[i])); out_r.append(int(region[i]))
            src = buf_src.get(i)
            if src is not None and src in live_addr and live_addr[src] not in early:
                sid = live_addr[src]
                early.add(sid)
                out_a.append(0); out_s.append(newsize.get(sid, 0)); out_id.append(sid); out_r.append(int(region[i]))
        else:
            a = int(aid[i])
            if live_addr.get(int(addr[i])) == a:
                del live_addr[int(addr[i])]
            if a in early:
                continue
            out_a.append(0); out_s.append(newsize.get(a, int(size[i]))); out_id.append(a); out_r.append(int(region[i]))
    new_index_of_old.append(len(out_a))
    return ({"action": np.asarray(out_a, np.int8), "size": np.asarray(out_s, np.int64),
             "alloc_id": np.asarray(out_id, np.int64), "region": np.asarray(out_r, np.int16)},
            {"early_freed_downcast_sources": len(early)}, new_index_of_old)
