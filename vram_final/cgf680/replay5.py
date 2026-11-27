"""Replay v5 (2026-11-23): replay3.replay + cuDNN plan fallback under memory pressure. replay3 is not changed.

Stream entries: action 1 = alloc, 0 = free (as replay3), 2 = cuDNN workspace request of a conv call: size = index of a
conv key in `cands` (list of candidate workspace sizes in heuristic order, measured by gpu_probe/cudnn_ws_probe.py).
PyTorch 2.6 (Conv_v8.cpp, benchmark=False) behaviour modelled:
  - the plan chosen for a key is cached; the next call of the same key tries the cached plan first;
  - the workspace is allocated through the caching allocator (which, on failure, releases cached free segments and
    retries) right after the output and freed when the call returns;
  - if the workspace allocation of the cached plan fails, plans are tried again in heuristic order from the next one
    (the first plan whose workspace can be allocated wins and is cached). A candidate list ending in a measured
    end state 'zero' ends with ws 0; other end states (small_pool_limit / shared_segment_limit / larger_fallback /
    max_candidates) leave the tail unknown -> the last measured candidate is final (reported in out['fallback']).
With enough memory every call uses candidate 1, so the result equals replay3 with candidate-1 workspaces inserted.
"""
from cgf680.replay3 import CUBLAS_WS, RUN_NAMES, new_allocator


def replay(st, cap, cands, policy="descending", ws_rel=None, ws_bytes=CUBLAS_WS, fast=True):
    sim, dev, stream = new_allocator(cap, policy)
    seg_ch = sim._trace.max_segment_changes
    use_ch = sim._trace.max_usage_changes
    live, chosen, fb = {}, {}, {"fallbacks": 0, "keys": {}, "exhausted": 0}
    out = {"cap": int(cap), "policy": policy, "ws_rel": ws_rel, "oom": None, "fallback": fb}
    runs, phase = [], "load"

    def conv_ws(k):
        c = cands[k]
        i = chosen.get(k, 0)
        while True:
            w = c[i]
            if w == 0:
                chosen[k] = i
                return
            try:
                b = sim.malloc(dev, stream, w)
            except MemoryError:
                if i + 1 >= len(c):
                    fb["exhausted"] += 1
                    raise
                i += 1
                fb["fallbacks"] += 1
                fb["keys"][k] = i
                continue
            chosen[k] = i
            sim.free(b)
            return

    try:
        for a, s, k in st["load"]:
            if a:
                live[("L", k)] = sim.malloc(dev, stream, s)
            else:
                b = live.pop(("L", k), None)
                if b is not None:
                    sim.free(b)
        out["after_load_reserved"] = sim.total_allocated_memory
        act, size, aid = st["act"], st["size"], st["aid"]
        for r, name in enumerate(RUN_NAMES):
            phase = name
            n_seg0, fb0 = len(seg_ch), fb["fallbacks"]
            res0 = sim.total_allocated_memory
            peak_r, peak_a = res0, use_ch[-1]
            start = {"reserved": res0, "allocated": use_ch[-1]}
            for j in range(len(act)):
                if r == 0 and ws_rel is not None and j == ws_rel:
                    live[("WS", 0)] = sim.malloc(dev, stream, ws_bytes)
                if act[j] == 2:
                    n_use = len(use_ch)
                    conv_ws(size[j])
                    if sim.total_allocated_memory > peak_r:
                        peak_r = sim.total_allocated_memory
                    if len(use_ch) > n_use and max(use_ch[n_use:]) > peak_a:   # workspace counts in the allocated peak
                        peak_a = max(use_ch[n_use:])
                    continue
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
                         "end_reserved": sim.total_allocated_memory, "segment_events": seg_events,
                         "fallbacks": fb["fallbacks"] - fb0})
            if fast and r >= 1 and seg_events == 0 and fb["fallbacks"] == fb0 and r < len(RUN_NAMES) - 1:
                for name2 in RUN_NAMES[r + 1:]:
                    runs.append({**runs[-1], "run": name2, "repeated": True})
                break
    except MemoryError as exc:
        out["oom"] = {"run": phase, "reserved": sim.total_allocated_memory, "allocated": use_ch[-1], "error": str(exc)[:200]}
        return out
    m0 = runs[-1]
    out.update(runs=runs, m0_start_reserved=m0["start"]["reserved"], m0_peak_reserved=m0["peak"]["reserved"],
               m0_start_allocated=m0["start"]["allocated"], m0_peak_allocated=m0["peak"]["allocated"],
               max_reserved=max(seg_ch), max_allocated=max(use_ch))
    return out
