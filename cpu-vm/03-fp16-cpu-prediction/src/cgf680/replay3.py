"""Reserved prediction v3: replay a stored CPU request stream (invocation_fp16_requests.npz) into xMem.

Differences from the v2 replay in cpu_worker2.replay_protocol (v2 stays the production value, unchanged):
  * segment address policy `descending` (each new segment placed below all previous ones). Block choice among
    equal-size free blocks uses (stream, size, address); the stock xMem GPUDevice model numbers addresses
    differently from the CUDA driver, which made replays of GPU request streams diverge by up to +896 MiB.
    INUGPU 2026-11-16 guide §3.3; reproduced on INUOS for 12/12 GPU traces.
  * GPU-only cuBLAS workspace request (8,519,680 B, never freed) inserted in warmup_0 right after the output request
    of the first GEMM-type call (position per model from cgf680.gemm_probe, relative to the invocation start).
  * capacity: the allocator-usable GPU capacity (cuda_total_memory - memory outside the allocator); OOM-retry of the
    caching allocator (release cached segments, retry) is then reproduced by xMem itself.
  * outputs R_at (measure_0 start reserved at a capacity), R_inf (capacity 1 TiB), R_min (smallest capacity, 2 MiB
    resolution, that replays without OOM).
  * light trace: only the reserved/allocated counters are kept (the stock xMem Trace copies the full segment list on
    every event, which needs >28 GB for large T5 streams). Allocation decisions never read the trace.
The xMem source is not modified (pinned commit, lock-verified by callers); the address policy is set per instance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CUBLAS_WS = 8_519_680          # default only; per target GPU use cgf680.gpu_rules.cublas_workspace()
MIB = 1 << 20
CAP_INF = 1 << 40
POLICIES = ("xmem", "descending")
RUN_NAMES = ("warmup_0", "warmup_1", "warmup_2", "measure_0")


def _xmem():
    import logging
    logging.disable(logging.WARNING)     # xMem logs every OOM attempt (R_min bisection) at WARNING
    from perf_estimator.allocator.allocator import CachingAllocator, Device, Stream, Trace

    class LightTrace(Trace):
        def _record(self, block, action, use_size=True):
            pass
    return CachingAllocator, Device, Stream, LightTrace


class _Addr:
    __slots__ = ("address",)

    def __init__(self, a):
        self.address = a


def new_allocator(cap, policy="descending"):
    if policy not in POLICIES:
        raise ValueError(policy)
    CachingAllocator, Device, Stream, LightTrace = _xmem()
    sim = CachingAllocator(allowed_memory_maximum=int(cap))
    sim._trace = LightTrace(sim._segments)
    if policy == "descending":
        state = [1 << 46]

        def gpu_malloc(size):
            state[0] -= size
            return _Addr(state[0])
        sim._gpu_device.malloc = gpu_malloc
        sim._gpu_device.addr_free = lambda addr: None
    return sim, Device(index=0), Stream(index=0)


def load_stream(case_dir):
    """-> dict(load=[(is_alloc, size, id)], act, size, aid (python lists), m4_local)"""
    z = np.load(Path(case_dir) / "invocation_fp16_requests.npz")
    le = z["load_events"]
    return {"load": [(int(a), int(s), int(k)) for a, s, k in le.tolist()],
            "act": z["action"].astype(np.int64).tolist(), "size": z["size"].astype(np.int64).tolist(),
            "aid": z["alloc_id"].astype(np.int64).tolist(), "m4_local": int(z["m4_local"])}


def replay(st, cap, policy="descending", ws_rel=None, ws_bytes=CUBLAS_WS, fast=True):
    """Load -> warmup x3 -> measure_0. ws_rel: index inside the invocation stream before which the workspace request
    is inserted in warmup_0 (None: no workspace). fast: once a run k>=1 creates/releases no segment, the remaining runs
    are identical (all invocation blocks are freed and coalesced at run end), so they are not replayed."""
    sim, dev, stream = new_allocator(cap, policy)
    seg_ch = sim._trace.max_segment_changes
    use_ch = sim._trace.max_usage_changes
    live = {}
    out = {"cap": int(cap), "policy": policy, "ws_rel": ws_rel, "oom": None}
    runs, phase = [], "load"
    try:
        for a, s, k in st["load"]:
            if a:
                live[("L", k)] = sim.malloc(dev, stream, s)
            else:
                b = live.pop(("L", k), None)
                if b is not None:
                    sim.free(b)
        out["after_load_reserved"] = sim.total_allocated_memory
        out["after_load_allocated"] = use_ch[-1]
        act, size, aid = st["act"], st["size"], st["aid"]
        for r, name in enumerate(RUN_NAMES):
            phase = name
            n_seg0 = len(seg_ch)
            res0 = sim.total_allocated_memory
            peak_r, peak_a = res0, use_ch[-1]
            start = {"reserved": res0, "allocated": use_ch[-1]}
            for j in range(len(act)):
                if r == 0 and ws_rel is not None and j == ws_rel:
                    live[("WS", 0)] = sim.malloc(dev, stream, ws_bytes)
                if act[j]:
                    live[(r, aid[j])] = sim.malloc(dev, stream, size[j])
                    if sim.total_allocated_memory > peak_r:
                        peak_r = sim.total_allocated_memory
                    if use_ch[-1] > peak_a:
                        peak_a = use_ch[-1]
                else:
                    b = live.pop((r, aid[j]), None)
                    if b is not None:
                        sim.free(b)
            seg_events = len(seg_ch) - n_seg0
            runs.append({"run": name, "start": start, "peak": {"reserved": peak_r, "allocated": peak_a},
                         "end_reserved": sim.total_allocated_memory, "segment_events": seg_events})
            if fast and r >= 1 and seg_events == 0 and r < len(RUN_NAMES) - 1:
                for name2 in RUN_NAMES[r + 1:]:
                    runs.append({"run": name2, "start": dict(runs[-1]["start"]), "peak": dict(runs[-1]["peak"]),
                                 "end_reserved": runs[-1]["end_reserved"], "segment_events": 0, "repeated": True})
                break
    except MemoryError as exc:
        out["oom"] = {"run": phase,
                      "reserved": sim.total_allocated_memory, "allocated": use_ch[-1], "error": str(exc)[:200]}
        return out
    m0 = runs[-1]
    out.update(runs=runs, m0_start_reserved=m0["start"]["reserved"], m0_peak_reserved=m0["peak"]["reserved"],
               m0_start_allocated=m0["start"]["allocated"], m0_peak_allocated=m0["peak"]["allocated"],
               max_reserved=max(seg_ch), max_allocated=max(use_ch),
               retry_released=sum(1 for i in range(1, len(seg_ch)) if seg_ch[i] < seg_ch[i - 1]))
    return out


def r_min(st, policy, ws_rel, lo, hi, step=2 * MIB, fast=True, ws_bytes=CUBLAS_WS):
    """Smallest capacity (multiple of `step`) in (lo, hi] whose replay finishes without OOM (bisection; assumes
    monotonicity, as INUGPU capacity_sweep.py). hi must replay without OOM."""
    lo, hi = lo // step, -(-hi // step)
    n = 0
    if hi - 1 > lo:                      # common case R_min == R_inf: one replay settles it
        n += 1
        if replay(st, (hi - 1) * step, policy, ws_rel, ws_bytes, fast=fast)["oom"]:
            return hi * step, n
        hi -= 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        n += 1
        if replay(st, mid * step, policy, ws_rel, ws_bytes, fast=fast)["oom"]:
            lo = mid
        else:
            hi = mid
    return hi * step, n


def predict(case_dir, caps, policy="descending", ws_rel=None, want_rmin=True, fast=True, ws_bytes=CUBLAS_WS, st=None):
    """caps: {name: bytes}. ws_bytes: cuBLAS workspace of the target GPU (cgf680.gpu_rules). Returns R_inf, R_at[name]
    (+ oom flag), R_min."""
    st = st or load_stream(case_dir)
    inf = replay(st, CAP_INF, policy, ws_rel, ws_bytes, fast=fast)
    out = {"policy": policy, "ws_rel": ws_rel, "ws_bytes": ws_bytes if ws_rel is not None else 0, "fast": fast,
           "R_inf": inf["m0_start_reserved"], "R_inf_m0_peak": inf["m0_peak_reserved"],
           "R_inf_max_reserved": inf["max_reserved"], "max_allocated": inf["max_allocated"],
           "after_load_reserved": inf["after_load_reserved"], "R_at": {}}
    for name, cap in caps.items():
        if inf["max_reserved"] <= cap:
            out["R_at"][name] = {"reserved": inf["m0_start_reserved"], "peak": inf["m0_peak_reserved"], "oom": False,
                                 "retry": False, "same_as_inf": True}
            continue
        r = replay(st, cap, policy, ws_rel, ws_bytes, fast=fast)
        out["R_at"][name] = ({"reserved": None, "oom": True, "oom_at": r["oom"]} if r["oom"] else
                             {"reserved": r["m0_start_reserved"], "peak": r["m0_peak_reserved"], "oom": False,
                              "retry": r["retry_released"] > 0})
    if want_rmin:
        out["R_min"], out["R_min_bisect_steps"] = r_min(st, policy, ws_rel, inf["max_allocated"],
                                                        inf["max_reserved"], fast=fast, ws_bytes=ws_bytes)
    return out
