"""FP16 shadow emulation, rule v2 (2026-11-16). The production rule (cgf680.shadow) stays unchanged.

Change: kernel-internal temporaries (traced requests not matched to any torch-function output) inherit the FP16-run
tag of the call that was executing. Rule v1 took the tag of that call's OUTPUT. When the output is not floating point
(e.g. torch.isinf -> bool, computed inside ATen as abs(x) == inf), the floating temporary abs(x) was kept at FP32 size
('x' = exact). Rule v2 uses the tag of the call's floating INPUTS in that case ('f' if any input is fixed fp32, else
'h'); calls without floating inputs keep 'x'. Found on T5 (T5Block FP16 inf clamp: 24 MiB GPU vs 48 MiB CPU, 24x per
warmup in t5-base b128/128/512), see results/reserved_v3/confirm/T5_DOUBLE_SIZE_KO.md.

(The file name ends in "shadow.py" on purpose: shadow._caller_frame skips frames of such files when it reads the
calling source line.)
"""
from __future__ import annotations

from cgf680.shadow import ShadowMode, _tensors, match_records, shadow_of

RULE = "v2_internal_temp_from_float_inputs"


class ShadowModeV2(ShadowMode):
    """Same tagging as ShadowMode; additionally remembers, per call number, the tag of the call's floating inputs."""

    def __init__(self):
        super().__init__()
        self.call_in_tag = {}

    def __torch_function__(self, func, types, args=(), kwargs=None):
        before = self.calls
        fl = None
        if self.enabled:
            ins = _tensors(list(args) + list((kwargs or {}).values()), [])
            tags = [shadow_of(t) for t in ins if t.is_floating_point()]
            if tags:
                fl = "f" if "f" in tags else "h"
        res = super().__torch_function__(func, types, args, kwargs)
        if self.calls != before and fl is not None:
            self.call_in_tag[self.calls] = fl
        return res


def match_records_v2(ev, records, call_in_tag):
    """match_records (unchanged matching) + rule v2 for unmatched requests whose executing call has a non-float output."""
    tags, buf_src, st = match_records(ev, records)
    act, addr, size = ev["action"], ev["addr"], ev["size"]
    # recover, for every matched alloc event, the call that produced it (same matching as match_records)
    from cgf680.gemm_probe import record_hits
    hits = record_hits(ev, records)             # {event index: (kind, call)}
    nxt_call = None
    changed = 0
    changed_bytes = 0
    for i in range(len(act) - 1, -1, -1):
        if act[i] != 1:
            continue
        if i in hits:
            if hits[i][0] != "buf":
                nxt_call = hits[i][1]
            continue
        if tags[i] == "inferred_x" and nxt_call is not None:
            it = call_in_tag.get(nxt_call)
            if it in ("h", "f"):
                tags[i] = "v2_inferred_" + it          # ends in h/f: emulate_fp16 halves h
                changed += 1
                changed_bytes += int(size[i])
    st["rule"] = RULE
    st["rule_v2_retagged"] = changed
    st["rule_v2_retagged_bytes"] = changed_bytes
    return tags, buf_src, st

