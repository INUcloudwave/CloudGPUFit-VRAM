"""CPU request trace (warmup x3 + measure x1 in ONE profiler session) -> ordered xMem replay.

One case per process (fresh worker). Output directory layout (<out>/):
  progress.json    last stage/run (updated before each stage; read by the runner after a kill)
  mem.jsonl        VmRSS samples (1 s) with stage label
  events.npz       normalized request events (Ev Idx order): action, size, alloc_id, order_key, ts, region
  result.json      case result (status, prediction bytes, inputs, timings, provenance)
  trace.json       raw chrome trace (deleted after parsing unless --keep-raw-trace)

Replay rules (xMem pinned CachingAllocator via cgf_diag.replay.Replayer, strictly in Ev Idx order):
  * resident: unique storages of parameters+buffers (shared storage counted once) then device inputs,
    malloc'd in that order before the first traced request and never freed (existing xMem method).
  * request = CPU allocation inside a cgf680:: region (one region per invocation); its free is
    replayed wherever it occurs in the session (also between invocations).
  * boundary markers (distinct-size uint8 allocations outside regions) give exact positions in the
    event order: M0 session start, M1/M2 between warmups, M3 measure start, M4 measure end (output
    still referenced), M5 after the output is released.
  * allocator state built by the warmups is kept; measure peak is computed from M3 onward.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from cgf680 import protocol as P

MARK_BASE = 1_234_567  # marker k has MARK_BASE + 2*k bytes
MARKERS = ["M0_session_start", "M1_after_warmup0", "M2_after_warmup1", "M3_measure_start", "M4_measure_end_with_output",
           "M5_measure_end_output_released"]
CAPACITY_BYTES = 1 << 40  # 1 TiB: unconstrained (no capacity pressure); GPU OOM is classified separately


def now():
    return time.perf_counter()


class Progress:
    def __init__(self, out):
        self.path = Path(out) / "progress.json"
        self.state = {"stage": "init", "run": None}

    def set(self, stage, run=None, **extra):
        self.state = {"stage": stage, "run": run, "wall": time.time(), **extra}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state))
        tmp.replace(self.path)


def vmrss():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None


def start_rss_sampler(path, progress, stage_peaks):
    def loop():
        with open(path, "a") as f:
            while True:
                r = vmrss()
                st = progress.state.get("stage")
                if r is not None:
                    stage_peaks[st] = max(stage_peaks.get(st, 0), r)
                f.write(json.dumps({"t": round(time.time(), 2), "stage": st, "run": progress.state.get("run"), "rss": r}) + "\n")
                f.flush()
                time.sleep(1.0)
    threading.Thread(target=loop, daemon=True).start()


def unique_storages(tensors):
    seen, sizes = set(), []
    for t in tensors:
        s = t.untyped_storage()
        key = (s.data_ptr(), s.nbytes())
        if key in seen or s.nbytes() == 0:
            continue
        seen.add(key)
        sizes.append(s.nbytes())
    return sizes


def install_finite_hook(model, kind, state):
    """Finite check of every forward's logits through a numpy VIEW (numpy temporaries do not go
    through the c10 CPU allocator -> no traced requests are added). First failure is recorded."""
    def hook(module, args, output):
        logits = getattr(output, "logits", None)
        if logits is None or state["first_nonfinite"] is not None:
            return
        view = logits.detach()
        if kind != "vision_cls" and view.dim() == 3:
            view = view[:, -1, :]
        arr = view.numpy()
        if not np.isfinite(arr).all():
            state["first_nonfinite"] = {"run": state["run"], "forward_index_in_run": state["forwards"]}
        state["forwards"] += 1
    return model.register_forward_hook(hook)


def classify_exception(exc):
    msg = str(exc)
    if isinstance(exc, P.InputMismatch):
        return "input_mismatch"
    if isinstance(exc, P.TokenCountMismatch):
        return "input_mismatch"  # sub-reason token count, kept in error text
    if isinstance(exc, MemoryError) or "can't allocate memory" in msg or "DefaultCPUAllocator" in msg:
        return "cpu_ram_oom"
    if "not implemented for 'Half'" in msg or ("Half" in msg and "not implemented" in msg):
        return "fp16_unsupported"
    return "error"


class Session:
    """ONE kineto profiler session (CPU activity, profile_memory=True, no shapes/stacks).

    scope="user": RecordFunction callbacks limited to USER_SCOPE, i.e. only our cgf680:: region
    annotations are recorded as ops; [memory] events come from the allocator's memory reporting hook,
    which is independent of RecordFunction scopes, so the same allocation/free events are captured
    without millions of aten op events (RAM/disk bound). scope="all": every op (torch.profiler default).
    The low-level API is used so no Python FunctionEvent list is built at stop()."""

    def __init__(self, scope):
        import torch._C._profiler as PP
        from torch.autograd import ProfilerActivity
        self.scope = scope
        self.cfg = PP.ProfilerConfig(PP.ProfilerState.KINETO, False, True, False, False, False, PP._ExperimentalConfig())
        self.acts = {ProfilerActivity.CPU}
        self.scopes = {PP.RecordScope.USER_SCOPE} if scope == "user" else set()
        self.result = None

    def describe(self):
        return {"api": "torch.autograd._enable_profiler (kineto, low-level)", "activities": ["CPU"],
                "profile_memory": True, "record_shapes": False, "with_stack": False,
                "record_scopes": "USER_SCOPE only" if self.scope == "user" else "all", "sessions_per_worker": 1}

    def start(self):
        torch.autograd._prepare_profiler(self.cfg, self.acts)
        torch.autograd._enable_profiler(self.cfg, self.acts, self.scopes)

    def stop(self):
        self.result = torch.autograd._disable_profiler()

    def export(self, path):
        self.result.save(path)


# ----------------------------------------------------------------------------- trace parsing
def parse_trace(path, region_prefix="cgf680::"):
    """Stream the chrome trace; keep only region annotations and CPU [memory] events (compact)."""
    from inference_trace_runtime import iter_trace_events
    regions = []
    ts, idx, nbytes, addr = [], [], [], []
    for e in iter_trace_events(path):
        cat, name = e.get("cat"), e.get("name")
        if cat == "user_annotation" and str(name).startswith(region_prefix):
            regions.append((name, float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0))))
        elif name == "[memory]" and cat == "cpu_instant_event":
            a = e.get("args", {})
            if a.get("Device Type") == 0 and a.get("Bytes", 0) != 0:
                ts.append(float(e["ts"]))
                idx.append(int(a["Ev Idx"]) if a.get("Ev Idx") is not None else -1)
                nbytes.append(int(a["Bytes"]))
                addr.append(int(a["Addr"]))
    regions.sort(key=lambda r: r[1])
    mem = {"ts": np.asarray(ts, dtype=np.float64), "ev_idx": np.asarray(idx, dtype=np.int64),
           "bytes": np.asarray(nbytes, dtype=np.int64), "addr": np.asarray(addr, dtype=np.int64)}
    return regions, mem


def normalize(regions, mem, n_markers):
    """Ev Idx order; region filter for allocations; pair frees by address; locate markers."""
    order = np.lexsort((mem["ts"], mem["ev_idx"]))
    ts, ev, nb, ad = (mem[k][order] for k in ("ts", "ev_idx", "bytes", "addr"))
    begins = np.asarray([r[1] for r in regions]) if regions else np.zeros(0)
    ends = np.asarray([r[2] for r in regions]) if regions else np.zeros(0)
    ri = np.searchsorted(begins, ts, side="right") - 1
    inside = (ri >= 0) & (ts <= ends[np.clip(ri, 0, max(len(ends) - 1, 0))]) if len(begins) else np.zeros(len(ts), bool)
    region_of = np.where(inside, ri, -1)
    stats = {"memory_events": int(len(ts)), "ev_idx_missing": int((ev < 0).sum()),
             "ts_inversions_in_ev_idx_order": int((np.diff(ts) < 0).sum()) if len(ts) > 1 else 0,
             "skipped_alloc_outside_regions": 0, "unmatched_frees": 0, "addr_reuse_while_live": 0,
             "free_size_mismatch": 0, "marker_allocs": 0}
    marker_pos = {}
    live = {}
    out_action, out_size, out_aid, out_key, out_ts, out_region, out_addr = [], [], [], [], [], [], []
    next_id = 0
    marker_sizes = {MARK_BASE + 2 * k: k for k in range(n_markers)}
    for i in range(len(ts)):
        b = int(nb[i])
        if b > 0:
            if region_of[i] < 0:
                k = marker_sizes.get(b)
                if k is not None and k not in marker_pos:
                    marker_pos[k] = len(out_action)  # number of request events before this marker
                    stats["marker_allocs"] += 1
                else:
                    stats["skipped_alloc_outside_regions"] += 1
                continue
            a = int(ad[i])
            if a in live:
                stats["addr_reuse_while_live"] += 1
            live[a] = (next_id, b)
            out_action.append(1); out_size.append(b); out_aid.append(next_id)
            out_key.append(int(ev[i])); out_ts.append(float(ts[i])); out_region.append(int(region_of[i])); out_addr.append(a)
            next_id += 1
        else:
            a = int(ad[i])
            item = live.pop(a, None)
            if item is None:
                stats["unmatched_frees"] += 1  # frees of pre-session storages, markers, outside allocs
                continue
            if item[1] != -b:
                stats["free_size_mismatch"] += 1
            out_action.append(0); out_size.append(-b); out_aid.append(item[0])
            out_key.append(int(ev[i])); out_ts.append(float(ts[i])); out_region.append(int(region_of[i])); out_addr.append(a)
    stats["live_requests_at_session_end"] = len(live)
    stats["live_request_bytes_at_session_end"] = int(sum(v[1] for v in live.values()))
    stats["requests"] = int(sum(out_action))
    stats["frees"] = int(len(out_action) - sum(out_action))
    ev_arrays = {"action": np.asarray(out_action, np.int8), "size": np.asarray(out_size, np.int64),
                 "alloc_id": np.asarray(out_aid, np.int64), "order_key": np.asarray(out_key, np.int64),
                 "ts": np.asarray(out_ts, np.float64), "region": np.asarray(out_region, np.int16),
                 "addr": np.asarray(out_addr, np.int64)}
    return ev_arrays, marker_pos, stats


# ----------------------------------------------------------------------------- replay
def replay(ev, marker_pos, resident_sizes, region_names, capacity=CAPACITY_BYTES):
    from cgf_diag.replay import Replayer
    rp = Replayer(capacity, policy="xmem_pinned", keep_decisions=False)
    rp.place_resident(resident_sizes, phase="resident")
    snap = {"after_resident": {"reserved": rp.reserved, "allocated": rp.allocated}}
    by_pos = {}
    for k, pos in marker_pos.items():
        by_pos.setdefault(pos, []).append(k)
    seg = {}  # running peaks per segment between markers
    cur_seg = "before_M0"
    seg[cur_seg] = [rp.reserved, rp.allocated]
    action, size, aid, region = ev["action"], ev["size"], ev["alloc_id"], ev["region"]
    n = len(action)

    def hit_markers(pos):
        nonlocal cur_seg
        for k in sorted(by_pos.get(pos, [])):
            snap[MARKERS[k]] = {"reserved": rp.reserved, "allocated": rp.allocated}
            cur_seg = f"after_{MARKERS[k]}"
            seg[cur_seg] = [rp.reserved, rp.allocated]

    for i in range(n):
        hit_markers(i)
        r = int(region[i])
        row = {"seq": i, "action": "alloc" if action[i] else "free", "size": int(size[i]), "stream": 0,
               "alloc_id": int(aid[i]), "phase": region_names[r] if r >= 0 else "between_invocations", "invocation": None}
        if not rp.apply(row):
            break
        s = seg[cur_seg]
        res, alc = rp.reserved, rp.allocated
        if res > s[0]:
            s[0] = res
        if alc > s[1]:
            s[1] = alc
    hit_markers(n)
    return rp, snap, seg


def measure_stats(snap, seg):
    """start = M3; peak = max over [M3, M5]; end_with_output = M4; end_released = M5."""
    def get(name):
        return snap.get(name)
    st, e4, e5 = get("M3_measure_start"), get("M4_measure_end_with_output"), get("M5_measure_end_output_released")
    if not (st and e4):
        return None
    peak_r = max(seg.get("after_M3_measure_start", [0, 0])[0], seg.get("after_M4_measure_end_with_output", [0, 0])[0], st["reserved"])
    peak_a = max(seg.get("after_M3_measure_start", [0, 0])[1], seg.get("after_M4_measure_end_with_output", [0, 0])[1], st["allocated"])
    return {"allocated_start_bytes": st["allocated"], "allocated_peak_bytes": peak_a,
            "allocated_end_bytes": e4["allocated"], "allocated_end_after_output_release_bytes": e5["allocated"] if e5 else None,
            "reserved_start_bytes": st["reserved"], "reserved_peak_bytes": peak_r,
            "reserved_end_bytes": e4["reserved"], "reserved_end_after_output_release_bytes": e5["reserved"] if e5 else None,
            "end_definition": "end = M4: measure invocation returned, output tensor still referenced; "
                              "*_after_output_release = M5 after `del output`"}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--cases", default="/exp/manifest/cases.jsonl")
    ap.add_argument("--pins", default="/exp/manifest/model_pins.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--keep-raw-trace", action="store_true")
    ap.add_argument("--conv1d-colmajor", action="store_true",
                    help="CPU-only: give transformers Conv1D weights column-major strides (same shape/bytes) so the "
                         "FP16 addmm takes the transposed-B fast path; must be shown request-identical first")
    ap.add_argument("--expected-input-sha", default=None, help="JSON {name: sha256} from the GPU side")
    ap.add_argument("--emulate-fp16-from-fp32", action="store_true",
                    help="run the FP16 case with FP32 weights/compute and derive FP16 request sizes (cgf680.shadow)")
    ap.add_argument("--profiler-scope", choices=["user", "all"], default="user")
    ap.add_argument("--stop-after", choices=["trace", "parse"], default=None, help="debug")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prog = Progress(out)
    stage_peaks = {}
    start_rss_sampler(out / "mem.jsonl", prog, stage_peaks)
    t_all = now()
    case = next(c for c in map(json.loads, Path(a.cases).read_text().splitlines()) if c["case_id"] == a.case_id)
    pin = P.load_pins(a.pins)[case["model"]]
    res = {"case_id": case["case_id"], "case": case, "model": case["model"], "revision": pin["revision"],
           "weights_file": pin["weights_file"], "status": "running", "protocol": {
               "warmup_invocations": P.N_WARMUP, "measure_invocations": P.N_MEASURE, "seed": P.SEED,
               "profiler_sessions": 1, "comparison_target": "GPU first measured invocation",
               "inputs": "host-generated, moved to device once before warmup, same tensors every invocation",
               "output_release": "del output after each invocation (outside regions, inside session)",
               "validation_forward_before_measure": False, "inference_mode": True, "eval": True},
           "timings_s": {}, "trace_files": {}}
    timings = res["timings_s"]

    def fail(status, stage, run, exc=None):
        res["status"] = status
        res["failure"] = {"stage": stage, "run": run, "error_type": type(exc).__name__ if exc else None,
                          "error": str(exc)[:4000] if exc else None,
                          "traceback": traceback.format_exc()[-6000:] if exc else None}

    try:
        import transformers
        torch.set_num_threads(a.threads)
        torch.manual_seed(P.SEED)
        res["environment"] = {
            "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "numpy": np.__version__, "device": "cpu", "torch_threads": torch.get_num_threads(),
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "mkldnn_fp16_supported": bool(torch.ops.mkldnn._is_mkldnn_fp16_supported()),
            "image": os.environ.get("CGF_IMAGE"), "image_id": os.environ.get("CGF_IMAGE_ID"),
            "container_memory_limit": os.environ.get("CGF_MEM_LIMIT"), "cpus_limit": os.environ.get("CGF_CPUS"),
            "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTORCH_CUDA_ALLOC_CONF",
                                                    "CUDA_MODULE_LOADING", "HF_HUB_OFFLINE")}}
        from cgf_diag.replay import verify_xmem
        res["xmem"] = verify_xmem("/work/xMem", "/exp/provenance/xmem-allocator-lock.json")
        res["xmem"]["allocator_settings"] = {"policy": "xmem_pinned (upstream CachingAllocator, no edits)",
                                             "capacity_bytes": CAPACITY_BYTES, "replay_order": "Ev Idx, sequential"}

        prog.set("model_load")
        t = now()
        emu = a.emulate_fp16_from_fp32
        if emu and case["dtype"] != "fp16":
            raise ValueError("--emulate-fp16-from-fp32 needs an fp16 case")
        run_case = dict(case, dtype="fp32") if emu else case
        res["dtype_mode"] = "fp32_trace_emulating_fp16" if emu else "native"
        model, attn = P.load_model(run_case, pin, "cpu")
        timings["model_load"] = now() - t
        res["attention_implementation"] = attn
        cfg = model.config
        unsupported = P.check_shape_supported(case, cfg)
        if unsupported:
            fail("unsupported_shape", "model_load", None, ValueError(unsupported))
            raise SystemExit(0)
        if a.conv1d_colmajor:
            from transformers.pytorch_utils import Conv1D
            n = 0
            with torch.no_grad():
                for m in model.modules():
                    if isinstance(m, Conv1D):
                        w = m.weight.data
                        m.weight.data = w.t().contiguous().t()
                        n += 1
            res["cpu_compute_modification"] = {"conv1d_colmajor": True, "modules": n,
                                               "note": "same shape/bytes per parameter; strides only"}

        prog.set("input_prep")
        t = now()
        host = P.make_host_inputs(case, cfg)
        P.validate_inputs(case, host)
        res["inputs"] = P.input_record(host)  # protocol (FP16-case) inputs: join key with the GPU side
        if emu:
            inputs = {k: (v.to(torch.float32) if v.is_floating_point() else v) for k, v in host.items()}
            res["inputs_used_for_fp32_trace"] = P.input_record(inputs)
        else:
            inputs = {k: v.to("cpu") for k, v in host.items()}  # device == cpu: same tensors (no copy)
        if a.expected_input_sha:
            exp = json.loads(a.expected_input_sha)
            bad = {k: (res["inputs"][k]["sha256"], v) for k, v in exp.items() if res["inputs"].get(k, {}).get("sha256") != v}
            if bad:
                raise P.InputMismatch(f"input sha mismatch: {bad}")
        gen_kwargs = P.generate_kwargs(case, cfg) if case["kind"] != "vision_cls" else {}
        res["generation_config"] = P.resolved_generation_config(model, gen_kwargs) if gen_kwargs else None
        param_sizes = unique_storages(list(model.parameters()) + list(model.buffers()))
        input_sizes = unique_storages(list(inputs.values()))
        if emu:
            def half_sizes(ts):
                seen, out = set(), []
                for t in ts:
                    st = t.untyped_storage()
                    key = (st.data_ptr(), st.nbytes())
                    if key in seen or st.nbytes() == 0:
                        continue
                    seen.add(key)
                    out.append(st.nbytes() // 2 if t.is_floating_point() else st.nbytes())
                return out
            param_sizes_fp32, input_sizes_fp32 = param_sizes, input_sizes
            param_sizes = half_sizes(list(model.parameters()) + list(model.buffers()))
            input_sizes = half_sizes(list(inputs.values()))
        res["resident"] = {"parameter_buffer_storages": len(param_sizes), "parameter_buffer_bytes": int(sum(param_sizes)),
                           "input_storages": len(input_sizes), "input_bytes": int(sum(input_sizes)),
                           "dedup_rule": "unique (storage data_ptr, nbytes); tied weights counted once",
                           "placement": "params+buffers (module order) then inputs, malloc'd before first request, never freed",
                           "difference_vs_gpu_start": "GPU start state comes from from_pretrained->cuda loading "
                                                      "(its temporaries/caching may leave extra cached segments) and "
                                                      "input .to(cuda); here each storage is a fresh malloc in an empty "
                                                      "allocator, so segment layout/fragmentation can differ"}
        timings["input_prep"] = now() - t

        fstate = {"run": None, "forwards": 0, "first_nonfinite": None}
        hook = install_finite_hook(model, case["kind"], fstate)

        def mark(k):
            m = torch.empty(MARK_BASE + 2 * k, dtype=torch.uint8)
            del m

        shadow = None
        if emu:
            from cgf680.shadow import ShadowMode, patch_fp16_branches
            shadow = ShadowMode()
            res["fp16_branch_patches"] = patch_fp16_branches(model)
        runs = [("warmup", i) for i in range(P.N_WARMUP)] + [("measure", i) for i in range(P.N_MEASURE)]
        region_names = [f"cgf680::{j:02d}::{ph}::{i}" for j, (ph, i) in enumerate(runs)]
        run_info = []
        gc.collect()
        gc.disable()
        prof = Session(a.profiler_scope)
        res["profiler"] = prof.describe()
        t_sess = now()
        prof.start()
        try:
            mark(0)
            for j, (ph, i) in enumerate(runs):
                prog.set("trace_run", run=f"{ph}{i}")
                fstate["run"], fstate["forwards"] = f"{ph}{i}", 0
                tr = now()
                with torch.profiler.record_function(region_names[j]):
                    if shadow is not None:
                        with shadow:
                            output, produced = P.invoke(model, case, inputs, gen_kwargs)
                    else:
                        output, produced = P.invoke(model, case, inputs, gen_kwargs)
                dur = now() - tr
                expect = case["new_tokens"] or 0
                run_info.append({"run": f"{ph}{i}", "seconds_incl_profiler": dur, "actual_new_tokens_per_sample": produced,
                                 "output_shape": list(output.shape), "forwards": fstate["forwards"]})
                print("CGF_RUN " + json.dumps(run_info[-1]), flush=True)
                if produced != expect:
                    raise P.TokenCountMismatch(f"{ph}{i}: actual new tokens {produced} != {expect}")
                if ph == "measure":
                    mark(4)
                del output
                mark({0: 1, 1: 2, 2: 3}.get(j, 5) if ph == "warmup" else 5)
        finally:
            prof.stop()
            timings["trace_session_total"] = now() - t_sess
            hook.remove()
        res["runs"] = run_info
        timings["warmup_runs_incl_profiler"] = sum(r["seconds_incl_profiler"] for r in run_info if r["run"].startswith("warmup"))
        timings["measure_run_incl_profiler"] = sum(r["seconds_incl_profiler"] for r in run_info if r["run"].startswith("measure"))
        res["finite_check"] = {"method": "forward hook, numpy view of logits (last position for text), every forward",
                               "first_nonfinite": fstate["first_nonfinite"]}
        gc.enable()

        prog.set("trace_export")
        t = now()
        raw = out / "trace.json"
        prof.export(str(raw))
        del prof
        gc.collect()
        timings["trace_export"] = now() - t
        res["trace_files"]["raw_trace_bytes"] = raw.stat().st_size
        if a.stop_after == "trace":
            res["status"] = "debug_stopped"
            raise SystemExit(0)

        prog.set("trace_parse")
        t = now()
        regions, mem = parse_trace(raw)
        if [r[0] for r in regions] != region_names:
            raise RuntimeError(f"regions in trace {[r[0] for r in regions]} != expected {region_names}")
        ev, marker_pos, nstats = normalize(regions, mem, len(MARKERS))
        del mem
        if sorted(marker_pos) != list(range(len(MARKERS))):
            raise RuntimeError(f"markers found {sorted(marker_pos)} (expected all {len(MARKERS)})")
        res["trace_stats"] = nstats
        res["trace_stats"]["marker_positions"] = {MARKERS[k]: v for k, v in sorted(marker_pos.items())}
        res["trace_stats"]["regions"] = [{"name": n, "ts_begin": b, "ts_end": e} for n, b, e in regions]
        res["trace_stats"]["per_region_requests"] = {region_names[r]: int(((ev["region"] == r) & (ev["action"] == 1)).sum())
                                                     for r in range(len(region_names))}
        res["trace_stats"]["per_region_request_bytes"] = {
            region_names[r]: int(ev["size"][(ev["region"] == r) & (ev["action"] == 1)].sum()) for r in range(len(region_names))}
        np.savez_compressed(out / "events.npz", **ev, resident_sizes=np.asarray(param_sizes + input_sizes, np.int64),
                            marker_pos=np.asarray([marker_pos[k] for k in range(len(MARKERS))], np.int64))
        res["trace_files"]["events"] = str(out / "events.npz")
        timings["trace_parse"] = now() - t
        if not a.keep_raw_trace:
            raw.unlink()
            res["trace_files"]["raw_trace_deleted"] = True
        else:
            res["trace_files"]["raw_trace"] = str(raw)
        if a.stop_after == "parse":
            res["status"] = "debug_stopped"
            raise SystemExit(0)

        if emu:
            prog.set("shadow_match")
            t = now()
            from cgf680.shadow import match_records, emulate_fp16
            rp32, snap32, seg32 = replay(ev, marker_pos, param_sizes_fp32 + input_sizes_fp32, region_names)
            res["prediction_fp32_raw"] = measure_stats(snap32, seg32)
            del rp32
            tags, buf_src, mstats = match_records(ev, shadow.records)
            ev16, estats, idx_map = emulate_fp16(ev, tags, buf_src)
            marker_pos = {k: idx_map[v] for k, v in marker_pos.items()}
            res["fp16_emulation"] = {"method": "FP32 trace; per-request FP16 size from shadow dtype (h: model dtype -> /2, "
                                               "f: fixed fp32, x: non-float); no-op casts reproduced as real buffers",
                                     "torch_function_calls": shadow.calls, "emulated_casts": shadow.emulated,
                                     "undetermined_casts": shadow.undetermined_casts,
                                     "match": mstats, **estats}
            np.savez_compressed(out / "events_fp16_emulated.npz", **ev16,
                                tags=np.asarray([t or "" for t in tags]),
                                marker_pos=np.asarray([marker_pos[k] for k in range(len(MARKERS))], np.int64))
            ev = ev16
            del shadow.records
            timings["shadow_match"] = now() - t
        prog.set("xmem_replay")
        t = now()
        rp, snap, seg = replay(ev, marker_pos, param_sizes + input_sizes, region_names)
        timings["xmem_replay"] = now() - t
        summ = rp.summary()
        res["replay"] = {"snapshots": snap, "segments_peaks": {k: {"reserved": v[0], "allocated": v[1]} for k, v in seg.items()},
                         "session_peak_reserved_bytes": summ["peak_reserved_bytes"],
                         "session_peak_allocated_bytes": summ["peak_allocated_bytes"],
                         "unknown_frees": summ["unknown_frees"], "live_blocks_at_end": summ["live_blocks_at_end"],
                         "segments_at_end": summ["segments_at_end"], "oom": summ["oom"]}
        res["prediction"] = measure_stats(snap, seg)
        if summ["oom"]:
            fail("replay_capacity_oom", "xmem_replay", None, RuntimeError(json.dumps(summ["oom"])))
        elif fstate["first_nonfinite"] is not None:
            res["status"] = "nonfinite_output"
            res["failure"] = {"stage": "trace_run", "run": fstate["first_nonfinite"]["run"],
                              "error": f"nonfinite logits {fstate['first_nonfinite']}", "prediction_kept_for_diagnosis": True}
        else:
            res["status"] = "ok"
    except SystemExit:
        pass
    except BaseException as exc:  # noqa: BLE001
        st = prog.state
        fail(classify_exception(exc), st.get("stage"), st.get("run"), exc)
    finally:
        timings["total_worker"] = now() - t_all
        res["rss"] = {"max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      "stage_peak_rss_bytes": stage_peaks}
        tmp = out / "result.json.tmp"
        tmp.write_text(json.dumps(res, indent=1, default=str))
        tmp.replace(out / "result.json")
        prog.set("done", status=res["status"])
        print("CGF_RESULT " + json.dumps({"case_id": res["case_id"], "status": res["status"],
                                          "prediction": res.get("prediction")}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
