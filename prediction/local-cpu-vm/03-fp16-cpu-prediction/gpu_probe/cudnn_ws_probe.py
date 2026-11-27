#!/usr/bin/env python3
"""cuDNN convolution workspace probe (one GPU type, FP16, inference) -> per-architecture workspace table.

Why: with torch.backends.cudnn.benchmark=False (as in the VRAM measurement worker), PyTorch 2.6 asks the cuDNN
heuristic for a ranked list of execution plans and runs the first one whose workspace can be allocated through the
caching allocator; the workspace is allocated right after the conv output and freed when the call returns; the chosen
plan is cached per conv parameter key. On OOM of the workspace the next plan is tried (Conv_v8.cpp try_configs).
So per (conv shape, GPU architecture, cuDNN version) we need: the workspace of plan 1, and of the plans that follow
when less memory is free (fallback order). This script measures both without any model, per conv signature.

Method per signature (fresh tensors, contiguous NCHW FP16 input and weight, no bias):
  1. candidate 1: empty the cache, reset peak, run F.conv2d once; ws = peak_allocated - allocated_before - output.
     Also recorded: number and total bytes of all allocations made during the call (memory_stats deltas).
  2. fallback candidates: cap the allocator (set_per_process_memory_fraction) so that after the output segment only
     (segment size of the previous ws - 2 MiB) is left, run again (the cached plan OOMs -> heuristics again -> first
     plan that fits); repeat until ws == 0, the cap reaches the output itself, or --max-candidates.
     Resolution: candidates whose workspace rounds to the same allocator segment size as the previous one are skipped;
     workspaces <= 1 MiB (small pool, same 2 MiB segment as the weight) cannot be excluded by a cap, so the search
     stops there (end = "small_pool_limit"). `end` records why the candidate search stopped.
     v3 2026-11-22 (INUGPU proposal, accepted by INUOS): the cap uses the REAL segments created by the previous
     candidate (memory_snapshot before/after the call): limit = workspace segment - 2 MiB, plus the real output
     segment(s), instead of seg_size(measured ws) / seg_size(out_bytes) (the measured ws can include ~1 MiB of other
     allocations, so the old limit let the same plan fit again: 724/3253 "no_change" in v2 -> 3 in the proposal run).
     A workspace placed in an already existing segment (ws_seg == 0, typically 1-10 MiB in the weight's 20 MiB
     segment) cannot be excluded by a cap: end = "shared_segment_limit" (normal stop, like small_pool_limit).
     end = "larger_fallback": the next plan needs a larger workspace (the search stops there).
     Fix 2026-11-20 (INUGPU trial): empty_cache before x, w are allocated - otherwise x lands in a cached segment of
     the previous signature, the cap leaves room inside it, and the same plan runs again (end = "no_change").
     Note: a capped run leaves the fallback plan in PyTorch's per-shape plan cache; signatures are unique, so
     candidate 1 of every signature is still measured with an empty cache entry.
  3. repeat check (--repeat): candidate 1 measured again in a fresh plan-cache state is not possible from Python, so
     the first measurement is repeated with the cached plan (must give the same ws).
Settings match the measurement worker: cudnn.benchmark=False, deterministic=False, TF32 off, CUDA_MODULE_LOADING=LAZY.

usage: python cudnn_ws_probe.py --signatures conv_signatures.json --out probe_<gpu>.jsonl [--src model] [--limit N]
Output: JSONL, first line = environment, then one line per signature (resumable: finished signatures are skipped).
"""
import argparse
import json
import os
import platform
import time

import torch
import torch.nn.functional as F

MIB = 1 << 20
KEYS = ("N", "C", "H", "W", "K", "R", "stride", "pad", "dil", "groups")


def seg_size(n):
    """Caching allocator segment size for one request of n bytes (PyTorch 2.x defaults, no expandable segments)."""
    n = max(512, (n + 511) // 512 * 512)
    if n <= MIB:
        return 2 * MIB
    if n < 10 * MIB:
        return 20 * MIB
    return (n + 2 * MIB - 1) // (2 * MIB) * (2 * MIB)


def stats():
    s = torch.cuda.memory_stats()
    return s["allocation.all.allocated"], s["allocated_bytes.all.allocated"]


def run_once(x, w, sg, total, cap=None):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if cap is not None:
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap / total))
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    n0, b0 = stats()
    segs0 = {s["address"] for s in torch.cuda.memory_snapshot()}
    try:
        y = F.conv2d(x, w, None, sg["stride"], sg["pad"], sg["dil"], sg["groups"])
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.set_per_process_memory_fraction(1.0)
        return {"oom": True, "error": str(e)[:160]}
    finally:
        if cap is not None:
            torch.cuda.set_per_process_memory_fraction(1.0)
    peak = torch.cuda.max_memory_allocated()
    n1, b1 = stats()
    new = [s for s in torch.cuda.memory_snapshot() if s["address"] not in segs0]
    ws_segs = sorted((s["total_size"] for s in new if s["allocated_size"] == 0), reverse=True)
    out_seg = sum(s["total_size"] for s in new if s["allocated_size"] > 0)   # real segment(s) holding the output
    out = y.untyped_storage().nbytes()
    out_r = (out + 511) // 512 * 512
    del y
    return {"oom": False, "ws": int(peak - base - out_r), "out_bytes": int(out), "n_allocs": int(n1 - n0),
            "alloc_bytes_extra": int(b1 - b0 - out_r),
            "ws_seg": int(ws_segs[0]) if ws_segs else 0, "new_free_segs": [int(x) for x in ws_segs], "out_seg": int(out_seg)}


def probe(sg, total, max_cand, repeat):
    dev = "cuda"
    torch.cuda.empty_cache()   # x, w get their own segments (not a cached segment left by the previous signature)
    x = torch.randn(sg["N"], sg["C"], sg["H"], sg["W"], device=dev, dtype=torch.float16)
    w = torch.randn(sg["K"], sg["C"] // sg["groups"], sg["R"], sg["R"], device=dev, dtype=torch.float16)
    r = {k: sg[k] for k in KEYS}
    r["src"] = sg.get("src")
    t0 = time.time()
    c1 = run_once(x, w, sg, total)
    r["candidates"] = [c1]
    if c1["oom"]:
        r["status"] = "oom_unlimited"
        return r
    if repeat:
        r["repeat_ws"] = run_once(x, w, sg, total)["ws"]
    prev = c1["ws"]
    r["end"] = "zero" if prev == 0 else None
    while prev > 0 and len(r["candidates"]) < max_cand:
        if prev <= MIB:          # small-pool request: shares a 2 MiB segment with w, a cap cannot exclude it
            r["end"] = "small_pool_limit"
            break
        last = r["candidates"][-1]
        if last.get("ws_seg", 0) == 0:   # workspace placed in an already existing segment (e.g. the 20 MiB one of w)
            r["end"] = "shared_segment_limit"
            break
        torch.cuda.empty_cache()
        live = torch.cuda.memory_reserved()
        limit = last["ws_seg"] - 2 * MIB  # INUGPU proposal 2026-11-22: real workspace segment, not seg_size(measured ws)
        cap = live + c1["out_seg"] + max(limit, 0)   # INUGPU proposal: real output segment(s), not seg_size(out_bytes)
        c = run_once(x, w, sg, total, cap=cap)
        c["cap_ws_limit"] = int(max(limit, 0))
        r["candidates"].append(c)
        if c["oom"]:
            r["end"] = "oom"
            break
        if c["ws"] > prev:          # cuDNN fallback order is not sorted by workspace size (3/3253 on RTX 3090)
            r["end"] = "larger_fallback"
            break
        if c["ws"] == prev:
            r["end"] = "no_change"
            break
        prev = c["ws"]
        if prev == 0:
            r["end"] = "zero"
    if r["end"] is None:
        r["end"] = "max_candidates"
    r["status"] = "ok"
    r["t_s"] = round(time.time() - t0, 3)
    del x, w
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signatures", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=None, help="only signatures whose src starts with this (model / grid / grid_dw)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-candidates", type=int, default=6)
    ap.add_argument("--mem-fraction", type=float, default=0.45, help="skip signatures whose x+w+out exceed this share")
    ap.add_argument("--no-repeat", action="store_true")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sigs = json.load(open(a.signatures))["signatures"]
    if a.src:
        sigs = [s for s in sigs if (s.get("src") or "").startswith(a.src)]
    if a.limit:
        sigs = sigs[:a.limit]
    total = torch.cuda.get_device_properties(0).total_memory
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            d = json.loads(line)
            if "N" in d:
                done.add(tuple(d[k] for k in KEYS))
    f = open(a.out, "a")
    if not done:
        p = torch.cuda.get_device_properties(0)
        f.write(json.dumps({"environment": {
            "gpu_name": p.name, "capability": list(torch.cuda.get_device_capability(0)), "total_memory": total,
            "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "cuda_module_loading": os.environ.get("CUDA_MODULE_LOADING"), "python": platform.python_version(),
            "image_id": os.environ.get("IMAGE_ID"), "label": os.environ.get("GPU_LABEL"),
            "settings": {"benchmark": False, "deterministic": False, "tf32": False, "dtype": "float16",
                         "memory_format": "contiguous NCHW", "bias": False}}}) + "\n")
    t0, n = time.time(), 0
    for sg in sigs:
        key = tuple(sg[k] for k in KEYS)
        if key in done:
            continue
        el = 2 * (sg["N"] * sg["C"] * sg["H"] * sg["W"] + sg["K"] * (sg["C"] // sg["groups"]) * sg["R"] ** 2)
        ho = (sg["H"] + 2 * sg["pad"] - sg["dil"] * (sg["R"] - 1) - 1) // sg["stride"] + 1
        el += 2 * sg["N"] * sg["K"] * ho * ho
        if el > a.mem_fraction * total:
            r = {k: sg[k] for k in KEYS}
            r.update(src=sg.get("src"), status="skipped_memory", bytes_needed=el)
        else:
            try:
                r = probe(sg, total, a.max_candidates, not a.no_repeat)
            except Exception as e:  # noqa: BLE001
                torch.cuda.set_per_process_memory_fraction(1.0)
                r = {k: sg[k] for k in KEYS}
                r.update(src=sg.get("src"), status="error", error=f"{type(e).__name__}: {str(e)[:200]}")
        f.write(json.dumps(r) + "\n")
        f.flush()
        n += 1
        if n % 100 == 0:
            print(f"{n} signatures, {time.time() - t0:.0f} s", flush=True)
    print(f"PROBE_DONE {n} signatures in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
