"""CPU worker v2 (protocol of INUGPU vram_worker.py): FP32 execution, FP16 request sizes by shadow dtype.

Per case (fresh container/process):
  1. load FP16 model (structure only) -> synthesize the GPU load-phase allocator requests (cgf680.load_emu,
     validated == RTX 3090 for 13/13 models), free it
  2. load FP32 model (default attention), GPU-style GenerationConfig, host inputs (seed 20261006)
  3. ONE profiler session (USER_SCOPE + [memory]) around ONE invocation (input device copy + generate/forward),
     markers: M0 before, M4 after (outputs+inputs alive), M5 after `del gi, o`
  4. FP16 sizes: shadow-dtype tags (h -> /2, f fixed fp32, x non-float) + reproduced no-op casts
  5. xMem (pinned CachingAllocator): load requests -> invocation trace x4 (warmup 0,1,2 + measure 0), cache kept;
     4 invocations were shown to have identical request sequences (36/36 cases) so one trace is replicated
  6. attention-kernel probe (separate small run after the session, full op scope) -> which aten ops run
Outputs use the GPU result keys (summary.first_measure_*, after_load).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from cgf680 import protocol2 as P
from cgf680.cpu_worker import (MARK_BASE, MARKERS, Progress, Session, normalize, parse_trace, start_rss_sampler)

CAP = 1 << 40
REGION = "cgf680::00::invoke::0"


def now():
    return time.perf_counter()


CUBLAS_WS = 4096 * 1024 * 2 + 16 * 1024 * 8   # 8.125 MiB per handle/stream (PyTorch 2.6, non-sm90)


def replay_protocol(load_events, seg, m4_local, n_warmup=P.N_WARMUP, gpu_workspace=0):
    """load events, then the single-invocation segment replayed (n_warmup + 1) times.
    gpu_workspace>0: VARIANT, a never-freed request of that size injected at the start of warmup_0 (GPU traces
    show the cuBLAS workspace allocated once, ~25-40 requests into warmup_0); not part of the CPU trace."""
    from cgf_diag.replay import Replayer
    rp = Replayer(CAP, policy="xmem_pinned", keep_decisions=False)
    i = 0
    load_peak_a = 0
    for a, s, aid in load_events:
        rp.apply({"seq": i, "action": "alloc" if a else "free", "size": s, "stream": 0, "alloc_id": ("L", aid),
                  "phase": "load", "invocation": None})
        load_peak_a = max(load_peak_a, rp.allocated)
        i += 1
    out = {"after_load": {"reserved": rp.reserved, "allocated": rp.allocated, "peak_allocated_during_load": load_peak_a,
                          "segments": len(rp.a._segments)}}
    act, size, aid = seg["action"], seg["size"], seg["alloc_id"]
    runs = []
    for k in range(n_warmup + 1):
        if k == 0 and gpu_workspace:
            rp.apply({"seq": i, "action": "alloc", "size": int(gpu_workspace), "stream": 0, "alloc_id": ("WS", 0),
                      "phase": "gpu_workspace_variant", "invocation": 0})
            i += 1
        st = {"reserved": rp.reserved, "allocated": rp.allocated}
        pr, pa = st["reserved"], st["allocated"]
        end = None
        for j in range(len(act)):
            if j == m4_local:
                end = {"reserved": rp.reserved, "allocated": rp.allocated}
            rp.apply({"seq": i, "action": "alloc" if act[j] else "free", "size": int(size[j]), "stream": 0,
                      "alloc_id": (k, int(aid[j])), "phase": f"run{k}", "invocation": k})
            i += 1
            r, al = rp.reserved, rp.allocated
            if r > pr:
                pr = r
            if al > pa:
                pa = al
        if end is None:
            end = {"reserved": rp.reserved, "allocated": rp.allocated}
        runs.append({"run": "measure_0" if k == n_warmup else f"warmup_{k}", "start": st,
                     "peak": {"reserved": pr, "allocated": pa}, "end_with_output": end,
                     "released": {"reserved": rp.reserved, "allocated": rp.allocated}, "segments_end": len(rp.a._segments)})
        if rp.oom:
            break
    summ = rp.summary()
    return out, runs, {"oom": summ["oom"], "unknown_frees": summ["unknown_frees"], "live_blocks_at_end": summ["live_blocks_at_end"]}


def attention_probe(model, case, inputs):
    """Which aten ops implement attention on this CPU (full op scope, B=1, same length, 2 new tokens)."""
    import torch._C._profiler as PP
    from torch.autograd import ProfilerActivity
    small = {k: v[:1].clone() for k, v in inputs.items()}
    cfg = PP.ProfilerConfig(PP.ProfilerState.KINETO, False, False, False, False, False, PP._ExperimentalConfig())
    torch.autograd._prepare_profiler(cfg, {ProfilerActivity.CPU})
    torch.autograd._enable_profiler(cfg, {ProfilerActivity.CPU}, set())
    try:
        with torch.inference_mode():
            if case["kind"] == "vision_cls":
                model(**small)
            else:
                g = model.generation_config
                old = (g.min_new_tokens, g.max_new_tokens)
                g.min_new_tokens = g.max_new_tokens = 2
                try:
                    model.generate(**small, generation_config=g)
                finally:
                    g.min_new_tokens, g.max_new_tokens = old
    finally:
        res = torch.autograd._disable_profiler()
    keys = ("attention", "sdp", "flash", "softmax", "bmm", "baddbmm", "efficient", "math")
    names = {}
    for e in res.events():
        n = e.name()
        if any(k in n.lower() for k in keys):
            names[n] = names.get(n, 0) + 1
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--cases", default="/exp/manifest/cases395.jsonl")
    ap.add_argument("--models", default="/exp/manifest/models.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--keep-raw-trace", action="store_true")
    ap.add_argument("--no-attn-probe", action="store_true")
    ap.add_argument("--dtype-mode", choices=("fp32_shadow", "fp16_real"), default="fp32_shadow",
                    help="fp32_shadow = production (FP32 execution, FP16 sizes by shadow dtype); fp16_real = model and inputs in FP16 on the CPU, no emulation (needs a CPU with FP16 support to be fast)")
    ap.add_argument("--shadow-rule", choices=("v1", "v2"), default="v1",
                    help="v1 = production FP16 size rule; v2 = internal temporaries of non-float-output calls take the float-input tag (cgf680.rule2_shadow)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prog = Progress(out)
    peaks = {}
    start_rss_sampler(out / "mem.jsonl", prog, peaks)
    t_all = now()
    case = next(c for c in map(json.loads, Path(a.cases).read_text().splitlines()) if c["case_id"] == a.case_id)
    pin = json.loads(Path(a.models).read_text())["models"][case["model"]]
    R = {"condition_id": case["case_id"], "case": case, "model": case["model"], "revision": pin["revision"],
         "status": "running", "failed_stage": None, "dtype_mode": None,
         "protocol": {"reference": "INUGPU vram_worker.py sha256 d06f0f15982c811c4bc6a24028f71728e461c6ee3651bcfd520ba49b0d675883",
                      "warmup": P.N_WARMUP, "compared_measure": "first measure (measure_0)",
                      "cpu_trace": "one invocation traced, replayed warmup x3 + measure x1 (identical requests, verified)",
                      "load_phase": "synthesized GPU from_pretrained requests (cgf680.load_emu)",
                      "attn_implementation_requested": "default (argument omitted)",
                      "generation_config": "GenerationConfig.from_model_config + update (hub generation_config unused)",
                      "inputs": "host seed 20261006; per invocation device copy (clone) kept until the record, then del"},
         "timings_s": {}, "files": {}}
    T = R["timings_s"]
    R["dtype_mode"] = "fp32_execution_fp16_request_sizes" if a.dtype_mode == "fp32_shadow" else "fp16_real_execution"
    EXEC = torch.float32 if a.dtype_mode == "fp32_shadow" else torch.float16

    def fail(status, exc=None):
        R["status"] = status
        R["failed_stage"] = prog.state.get("stage")
        if exc is not None:
            R["error"] = f"{type(exc).__name__}: {str(exc)[:3000]}"
            R["traceback"] = traceback.format_exc()[-5000:]

    try:
        import transformers
        torch.set_num_threads(a.threads)
        torch.manual_seed(P.SEED)
        R["environment"] = {"python": platform.python_version(), "torch": torch.__version__,
                            "transformers": transformers.__version__, "device": "cpu", "threads": torch.get_num_threads(),
                            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
                            "image": os.environ.get("CGF_IMAGE"), "image_id": os.environ.get("CGF_IMAGE_ID"),
                            "mem_limit": os.environ.get("CGF_MEM_LIMIT"), "cpuset": os.environ.get("CGF_CPUSET")}
        from cgf_diag.replay import verify_xmem
        R["xmem"] = verify_xmem("/work/xMem", "/exp/provenance/xmem-allocator-lock.json")

        # 1. load-phase synthesis from the FP16 model structure
        prog.set("model_load")
        t = now()
        from cgf680 import load_emu
        import glob
        m16 = P.load(case, pin, torch.float16)
        snap_dir = glob.glob(f"/exp/hf_cache/hub/models--{case['model'].replace('/', '--')}/snapshots/{pin['revision']}")[0]
        lseq, lstats = load_emu.synthesize(m16, snap_dir, pin["use_safetensors"])
        load_events, live = load_emu.to_events(lseq)
        R["load_synthesis"] = {**lstats, "events": len(load_events), "live_blocks": live}
        attn16 = getattr(m16.config, "_attn_implementation", None)
        del m16
        gc.collect()
        # 2. FP32 execution model
        m = P.finish_load(P.load(case, pin, EXEC), EXEC)
        T["model_load"] = now() - t
        cfg = m.config
        R["model_info"] = {"class": type(m).__name__, "attn_implementation": getattr(cfg, "_attn_implementation", None),
                           "attn_implementation_fp16_load": attn16, "attn_requested": "default"}
        bad = P.check_valid(case, cfg)
        if bad:
            raise P.InvalidInput(bad)
        prog.set("input_prep")
        host, sp = P.host_inputs(case, cfg)
        R["inputs"] = {k: {"shape": list(v.shape), "dtype": str(v.dtype), "sha256": P.sha_tensor(v)} for k, v in host.items()}
        R["special_tokens_replaced_by_3"] = sp
        inputs = {k: (v.to(EXEC) if v.is_floating_point() else v) for k, v in host.items()}
        if case["kind"] != "vision_cls":
            R["generation_config"] = P.generation_config(m, case).to_dict()
        from cgf680.shadow import ShadowMode, patch_fp16_branches, match_records, emulate_fp16
        import contextlib
        if a.dtype_mode == "fp16_real":
            from cgf680.gemm_shadow import FirstGemmRecorder
            R["fp16_branch_patches"] = "not needed (native FP16 execution)"
            shadow = FirstGemmRecorder()
        elif a.shadow_rule == "v2":
            from cgf680.rule2_shadow import ShadowModeV2, match_records_v2
            R["fp16_branch_patches"] = patch_fp16_branches(m)
            shadow = ShadowModeV2()
        else:
            R["fp16_branch_patches"] = patch_fp16_branches(m)
            shadow = ShadowMode()
        R["shadow_rule"] = a.shadow_rule

        # 3. one traced invocation
        def mark(k):
            x = torch.empty(MARK_BASE + 2 * k, dtype=torch.uint8)
            del x
        gc.collect()
        gc.disable()
        sess = Session("user")
        R["profiler"] = sess.describe()
        prog.set("trace_invocation")
        t = now()
        sess.start()
        try:
            mark(0)
            with torch.profiler.record_function(REGION):
                with shadow:
                    with torch.inference_mode():
                        gi, o, new = P.invoke(m, case, inputs)
            mark(4)
            del gi, o
            mark(5)
        finally:
            sess.stop()
            gc.enable()
        T["trace_invocation"] = now() - t
        R["new_tokens"] = new
        if case["kind"] != "vision_cls" and new != case["new_tokens"]:
            raise P.LengthMismatch(f"new tokens {new} != {case['new_tokens']}")

        prog.set("trace_export")
        t = now()
        raw = out / "trace.json"
        sess.export(str(raw))
        del sess
        gc.collect()
        T["trace_export"] = now() - t
        prog.set("trace_parse")
        t = now()
        regions, mem = parse_trace(raw)
        ev, mpos, nstats = normalize(regions, mem, len(MARKERS))
        del mem
        if [r[0] for r in regions] != [REGION] or not {0, 4, 5} <= set(mpos):
            raise RuntimeError(f"trace boundaries: regions={[r[0] for r in regions]} markers={sorted(mpos)}")
        R["trace_stats"] = {**nstats, "marker_positions": {MARKERS[k]: v for k, v in mpos.items()}}
        if not a.keep_raw_trace:
            raw.unlink()
        T["trace_parse"] = now() - t

        # 4. FP16 request sizes
        prog.set("shadow_match")
        t = now()
        if a.dtype_mode == "fp16_real":
            ev16 = {k: ev[k] for k in ("action", "size", "alloc_id", "region")}
            idx = list(range(len(ev["action"]) + 1))
            est, mst = {}, {"mode": "fp16_real: no emulation"}
            from cgf680.gemm_probe import record_hits
            hits = record_hits(ev, shadow.records)
            first = min(hits) if hits else None
            R["cublas_workspace_position"] = {"first_gemm_func": shadow.first_name, "ws_insert_rel":
                                              None if first is None else int(first - mpos[0] + 1),
                                              "rule": "right after the output request of the first GEMM-type call"}
        else:
            if a.shadow_rule == "v2":
                tags, buf_src, mst = match_records_v2(ev, shadow.records, shadow.call_in_tag)
            else:
                tags, buf_src, mst = match_records(ev, shadow.records)
            ev16, est, idx = emulate_fp16(ev, tags, buf_src)
        R["fp16_emulation"] = {"torch_function_calls": getattr(shadow, "calls", None), "emulated_casts": getattr(shadow, "emulated", None),
                               "undetermined_casts": getattr(shadow, "undetermined_casts", None), "match": mst, **est,
                               "fp32_request_bytes": int(ev["size"][ev["action"] == 1].sum()),
                               "fp16_request_bytes": int(ev16["size"][ev16["action"] == 1].sum())}
        del shadow.records
        p0, p4, p5 = idx[mpos[0]], idx[mpos[4]], idx[mpos[5]]
        seg = {k: v[p0:p5] for k, v in ev16.items()}
        np.savez_compressed(out / "invocation_fp16_requests.npz", **seg, m4_local=np.int64(p4 - p0),
                            load_events=np.asarray(load_events, dtype=np.int64))
        R["files"]["requests"] = str(out / "invocation_fp16_requests.npz")
        T["shadow_match"] = now() - t

        # 5. xMem replay: load -> warmup x3 -> measure
        prog.set("xmem_replay")
        t = now()
        al, runs, rinfo = replay_protocol(load_events, seg, p4 - p0)
        T["xmem_replay"] = now() - t
        R["after_load"] = al
        R["runs"] = runs
        R["replay"] = rinfo
        if rinfo["oom"]:
            raise RuntimeError(f"xmem replay oom at capacity {CAP}: {rinfo['oom']}")
        m0 = runs[-1]
        R["summary"] = {"first_measure_start_reserved": m0["start"]["reserved"],
                        "first_measure_start_allocated": m0["start"]["allocated"],
                        "first_measure_peak_reserved": m0["peak"]["reserved"],
                        "first_measure_peak_allocated": m0["peak"]["allocated"],
                        "first_measure_end_with_output_reserved": m0["end_with_output"]["reserved"],
                        "first_measure_end_with_output_allocated": m0["end_with_output"]["allocated"],
                        "first_measure_released_reserved": m0["released"]["reserved"],
                        "first_measure_released_allocated": m0["released"]["allocated"],
                        "after_load_reserved": al["after_load"]["reserved"],
                        "after_load_allocated": al["after_load"]["allocated"],
                        "load_peak_allocated": al["after_load"]["peak_allocated_during_load"]}
        _, runs_ws, _ = replay_protocol(load_events, seg, p4 - p0, gpu_workspace=CUBLAS_WS)
        w0 = runs_ws[-1]
        R["variant_with_cublas_workspace"] = {
            "note": "NOT the CPU prediction: same replay + one 8.125 MiB GPU-only cuBLAS workspace request at warmup_0 start",
            "first_measure_start_reserved": w0["start"]["reserved"], "first_measure_start_allocated": w0["start"]["allocated"],
            "first_measure_peak_reserved": w0["peak"]["reserved"], "first_measure_peak_allocated": w0["peak"]["allocated"]}
        R["finite_check"] = ("not_assessable (FP32 execution); use GPU validate result" if a.dtype_mode == "fp32_shadow"
                             else "not checked (fp16_real CPU run)")
        R["status"] = "ok"

        # 6. attention kernel probe (outside the traced session)
        if not a.no_attn_probe:
            prog.set("attn_probe")
            t = now()
            try:
                R["attention_ops_cpu"] = attention_probe(m, case, inputs)
            except Exception as exc:  # probe failure does not invalidate the prediction
                R["attention_ops_cpu"] = {"probe_error": f"{type(exc).__name__}: {exc}"[:500]}
            T["attn_probe"] = now() - t
    except P.InvalidInput as exc:
        fail("invalid_input", exc)
    except P.LengthMismatch as exc:
        fail("generation_length_mismatch", exc)
    except BaseException as exc:  # noqa: BLE001
        msg = str(exc)
        if isinstance(exc, MemoryError) or "can't allocate memory" in msg or "DefaultCPUAllocator" in msg:
            fail("cpu_oom", exc)
        elif "not implemented for 'Half'" in msg:
            fail("fp16_unsupported", exc)
        else:
            fail("compat_or_runtime_error", exc)
    finally:
        T["total_worker"] = now() - t_all
        R["rss"] = {"max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, "stage_peak_rss_bytes": peaks}
        tmp = out / "result.json.tmp"
        tmp.write_text(json.dumps(R, indent=1, default=str))
        tmp.replace(out / "result.json")
        prog.set("done", status=R["status"])
        print("CGF_RESULT " + json.dumps({"condition_id": R["condition_id"], "status": R["status"],
                                          "summary": R.get("summary")}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
