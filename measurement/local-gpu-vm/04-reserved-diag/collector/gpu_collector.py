#!/usr/bin/env python3
"""REFERENCE GPU collector for the CloudGPUFit reserved-error diagnosis (written on a CPU VM; NOT run on
a GPU yet). The inugpu operator may adapt paths/launch method, but must keep the recorded semantics
listed in GPU_COLLECTION_REQUEST.md. One invocation = one fresh process = one (case, run_id).

  CUDA_MODULE_LOADING=LAZY python gpu_collector_reference.py --manifest common_manifest.json \
      --case core__decoder_llm__small__fp16__01 --run-id R1_controlled --out OUT_DIR [--stacks python]

Outputs in OUT_DIR/<case>/<run_id>/: run_meta.json, boundaries.jsonl, invocations.jsonl,
memory_snapshot.pickle (original torch.cuda.memory._snapshot()), segments_<k>_<phase>.pickle.gz, collector.log
"""
import argparse
import copy
import ctypes
import gc
import gzip
import hashlib
import json
import os
import pickle
import platform
import sys
import time
import traceback
from pathlib import Path

if os.environ.get("CUDA_MODULE_LOADING") is None:
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"   # must precede CUDA initialisation
MODULE_LOADING_ENV_AT_START = os.environ.get("CUDA_MODULE_LOADING")

import torch  # noqa: E402

STAT_KEYS = ["allocated_bytes.all.current", "allocated_bytes.all.peak", "reserved_bytes.all.current", "reserved_bytes.all.peak",
             "active_bytes.all.current", "active_bytes.all.peak", "inactive_split_bytes.all.current", "requested_bytes.all.current",
             "requested_bytes.all.peak", "segment.all.current", "segment.large_pool.current", "segment.small_pool.current",
             "allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed",
             "num_alloc_retries", "num_ooms", "num_device_alloc", "num_device_free", "num_sync_all_streams",
             "oversize_allocations.current", "oversize_segments.current"]


def sha_tensor(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def module_loading_mode():
    """cuModuleGetLoadingMode: 1=EAGER, 2=LAZY (driver API, CUDA >= 11.7). None if unavailable."""
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        mode = ctypes.c_int(0)
        rc = lib.cuModuleGetLoadingMode(ctypes.byref(mode))
        return {"rc": rc, "mode": {1: "EAGER", 2: "LAZY"}.get(mode.value, mode.value)}
    except Exception as exc:  # report, do not guess
        return {"error": repr(exc)}


class Recorder:
    def __init__(self, out):
        self.out = out
        self.f = (out / "boundaries.jsonl").open("w")
        self.k = 0

    def mark(self, kind, phase=None, invocation=None, extra=None, segments=False):
        torch.cuda.synchronize()
        s = torch.cuda.memory_stats()
        row = {"k": self.k, "kind": kind, "phase": phase, "invocation": invocation, "time_us": time.time_ns() // 1000,
               "counters": {"alloc": s.get("allocation.all.allocated", 0), "free": s.get("allocation.all.freed", 0),
                            "segment_alloc": s.get("segment.all.allocated", 0), "segment_free": s.get("segment.all.freed", 0)},
               "memory_stats": {k: s.get(k) for k in STAT_KEYS}, **(extra or {})}
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()
        if segments:
            with gzip.open(self.out / f"segments_{self.k:03d}_{str(phase).replace('/', '_')}.pickle.gz", "wb") as g:
                pickle.dump(torch.cuda.memory_snapshot(), g)
        self.k += 1
        return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stacks", choices=["python", "all"], default="python")
    ap.add_argument("--max-entries", type=int, default=50_000_000)
    ap.add_argument("--threads", type=int, default=8)
    # [inugpu mod 1] optional suffix so that the optional stacks="all" run gets its own run directory/run_id
    ap.add_argument("--run-label-suffix", default="")
    a = ap.parse_args()
    manifest = json.loads(Path(a.manifest).read_text())
    case = next(c for c in manifest["cases"] if c["case_id"] == a.case)
    run = next(r for r in manifest["runs"] if r["run_id"] == a.run_id)
    run_label = a.run_id + a.run_label_suffix
    out = Path(a.out) / a.case / run_label
    out.mkdir(parents=True, exist_ok=False)  # never overwrite a previous run
    log = (out / "collector.log").open("w")

    def say(*x):
        print(*x, file=log, flush=True)
        print(*x, flush=True)

    # ---- process settings (identical to the previous worker where applicable)
    torch.set_num_threads(a.threads)
    torch.manual_seed(20261006)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1

    # ---- memory history BEFORE any CUDA allocation
    torch.cuda.memory._record_memory_history(enabled="all", context="all", stacks=a.stacks, max_entries=a.max_entries)
    torch.cuda.init()
    rec = Recorder(out)
    start = rec.mark("record_start")
    if start["memory_stats"]["reserved_bytes.all.current"] not in (0, None) or start["counters"]["alloc"]:
        say("WARNING: allocator not empty at record start")

    import transformers
    props = torch.cuda.get_device_properties(0)
    meta = {
        "case_id": a.case, "run_id": run_label, "manifest_run_id": a.run_id, "run_group": run["group"], "steps": run["steps"],
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(), "argv": sys.argv,
        "env": {k: os.environ.get(k) for k in ["CUDA_MODULE_LOADING", "PYTORCH_CUDA_ALLOC_CONF", "CUBLAS_WORKSPACE_CONFIG",
                                                "CUBLASLT_WORKSPACE_SIZE", "PYTORCH_NO_CUDA_MEMORY_CACHING", "CUDA_VISIBLE_DEVICES",
                                                "NVIDIA_VISIBLE_DEVICES", "OMP_NUM_THREADS"]},
        "cuda_module_loading_env_at_start": MODULE_LOADING_ENV_AT_START, "cuda_module_loading_driver_query": module_loading_mode(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "transformers": transformers.__version__, "python": platform.python_version(), "kernel": platform.release(),
        "gpu": {"name": props.name, "total_memory": props.total_memory, "uuid": str(getattr(props, "uuid", None)),
                "capability": list(torch.cuda.get_device_capability(0))},
        "allocator_backend": torch.cuda.get_allocator_backend(), "history": {"context": "all", "stacks": a.stacks, "max_entries": a.max_entries},
        "attention": "eager", "tf32": False, "seed": 20261006, "threads": a.threads,
    }
    # [inugpu mod 2] extra provenance (observed values only)
    meta["record_start_check"] = {"reserved_bytes.all.current": start["memory_stats"]["reserved_bytes.all.current"],
                                  "counters": start["counters"],
                                  "empty": start["memory_stats"]["reserved_bytes.all.current"] in (0, None)
                                  and not any(start["counters"].values())}
    meta["tf32_actual"] = {"matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                           "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32, "cudnn.benchmark": torch.backends.cudnn.benchmark}
    meta["host"] = {"cpu_count": os.cpu_count(), "torch_num_threads": torch.get_num_threads(), "hostname": platform.node()}
    meta["container"] = {k: os.environ.get(k) for k in ["CGF_IMAGE_ID", "CGF_IMAGE_TAG", "CGF_DOCKER_CMD", "CGF_HOST_LABEL"]}
    meta["weights"] = {"repo": case["model"], "revision": case["revision"], "use_safetensors": case["use_safetensors"]}
    meta["generation"] = case["generate_kwargs"]
    meta["batch_size"], meta["seq_len"], meta["max_new_tokens"], meta["dtype"] = (
        case["batch_size"], case["seq_len"], case["max_new_tokens"], case["dtype"])
    meta["differences_vs_bench_worker_v3"] = [
        "NVML sampler thread and idle/temperature wait omitted (no allocator activity)",
        "bench_worker_v3 also calls reset_peak_memory_stats() right before model_reuse in the second (memory) mode; "
        "this collector follows the request (reset only before model load and before each measure block). "
        "Affects peak statistics only, not allocator decisions or trace",
        "tokens/components modes omitted (run after memory mode in the original)",
        "memory history recording (_record_memory_history) and per-boundary synchronize()+memory_stats() added",
    ]
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        meta["gpu"].update(nvml_uuid=pynvml.nvmlDeviceGetUUID(h), nvml_total=pynvml.nvmlDeviceGetMemoryInfo(h).total,
                           driver=pynvml.nvmlSystemGetDriverVersion())
    except Exception as exc:
        meta["gpu"]["nvml_error"] = repr(exc)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str))

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[case["dtype"]]
    inv_log = (out / "invocations.jsonl").open("w")
    state = {}

    def model_load():
        kw = dict(case["model_load"]["kwargs"])
        kw["torch_dtype"] = dtype
        cls = getattr(transformers, case["model_load"]["class"])
        torch.cuda.reset_peak_memory_stats()
        model = cls.from_pretrained(case["model"], **kw).to(device="cuda", dtype=dtype).eval()
        model.config.output_attentions = False
        model.config.output_hidden_states = False
        state.update(model=model, config=copy.deepcopy(model.config), generation_config=copy.deepcopy(model.generation_config),
                     cpu_rng=torch.get_rng_state().clone(), cuda_rng=torch.cuda.get_rng_state().clone())
        cfg = model.config
        g = torch.Generator().manual_seed(20261006)
        ids = torch.randint(3, cfg.vocab_size, (case["batch_size"], case["seq_len"]), generator=g)
        for name in ("pad_token_id", "eos_token_id", "bos_token_id"):
            tok = getattr(cfg, name, None)
            if isinstance(tok, int):
                ids[ids == tok] = 3
        inputs = {"input_ids": ids, "attention_mask": torch.ones(case["batch_size"], case["seq_len"], dtype=torch.long)}
        got = sha_tensor(ids)
        if got != case["inputs"]["expected_input_ids_sha256"]:
            raise RuntimeError(f"input sha mismatch {got}")
        state["inputs"] = inputs
        state["pad"] = cfg.pad_token_id if cfg.pad_token_id is not None else cfg.eos_token_id
        # [inugpu mod 3] also check attention_mask sha and record it in run_meta
        got_mask = sha_tensor(inputs["attention_mask"])
        meta["input_ids_sha256"], meta["attention_mask_sha256"] = got, got_mask
        meta["input_sha_match"] = {"input_ids": got == case["inputs"]["expected_input_ids_sha256"],
                                   "attention_mask": got_mask == case["inputs"]["expected_attention_mask_sha256"]}
        meta["pad_token_id_used"] = state["pad"]
        meta["config_name_or_path"] = str(getattr(cfg, "_name_or_path", None))
        if got_mask != case["inputs"]["expected_attention_mask_sha256"]:
            raise RuntimeError(f"attention_mask sha mismatch {got_mask}")
        return {"input_ids_sha256": got, "attention_mask_sha256": got_mask}

    def finite_check():
        model, inputs = state["model"], state["inputs"]
        with torch.inference_mode():
            probe = model(**{n: v.to("cuda") for n, v in inputs.items()})
            finite = bool(torch.isfinite(probe.logits).all().item())
            del probe
        if not finite:  # [inugpu mod 4] bench_worker_v3 stops on nonfinite logits; do the same (record, then fail)
            state["failed_info"] = {"finite": finite}
            raise RuntimeError("nonfinite logits in finite_check")
        return {"finite": finite}

    def generate():
        model, inputs, n = state["model"], state["inputs"], case["max_new_tokens"]
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        with torch.inference_mode():
            gpu_inputs = {k: v.to("cuda") for k, v in inputs.items()}
            output = model.generate(**gpu_inputs, do_sample=False, num_beams=1, use_cache=True, min_new_tokens=n,
                                    max_new_tokens=n, pad_token_id=state["pad"], return_dict_in_generate=False,
                                    output_scores=False, output_attentions=False, output_hidden_states=False, streamer=None)
            host = output.cpu()
            new = output.shape[1] - case["seq_len"]
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        del output, gpu_inputs
        info = {"actual_new_tokens_per_sample": int(new), "expected_new_tokens": n, "output_shape": list(host.shape),
                "output_ids_sha256": sha_tensor(host), "elapsed_ms": (t1 - t0) / 1e6}
        if new != n:  # [inugpu mod 5] keep the observed values in invocations.jsonl before failing
            state["failed_info"] = info
            raise RuntimeError("actual output length differs from the condition")
        return info

    def between_modes():
        torch.cuda.synchronize()
        collected = gc.collect()          # keep_allocator_cache=True: NO empty_cache
        torch.cuda.synchronize()
        m = state["model"]
        m.config = copy.deepcopy(state["config"])
        m.generation_config = copy.deepcopy(state["generation_config"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        return {"gc_collected_objects": collected, "nvml_idle_wait": "omitted (no allocator activity)"}

    def empty_cache():
        rec.mark("empty_cache_before", "empty_cache", 0)
        torch.cuda.empty_cache()
        return {}

    fns = {"model_load": model_load, "finite_check": finite_check, "warmup": generate, "measure": generate,
           "between_modes": between_modes, "empty_cache": empty_cache}
    status = "ok"
    cur = None
    try:
        for phase, count in run["steps"]:
            step = phase.split("/")[-1]
            if step == "measure":
                torch.cuda.reset_peak_memory_stats()   # worker: before the measurement block (peak reset only)
            for i in range(count):
                cur = (phase, i)
                rec.mark("phase_begin", phase, i)
                info = fns[step]()
                rec.mark("phase_end", phase, i, segments=(i == count - 1))
                inv_log.write(json.dumps({"phase": phase, "invocation": i, "status": "ok", **info}) + "\n")
                inv_log.flush()
                say(phase, i, json.dumps(info))
    except torch.cuda.OutOfMemoryError:
        status = "gpu_oom"
        say(traceback.format_exc())
    except Exception:
        status = "error"
        say(traceback.format_exc())
    finally:
        if status != "ok" and cur is not None:  # [inugpu mod 6] failed invocation is recorded, not dropped
            inv_log.write(json.dumps({"phase": cur[0], "invocation": cur[1], "status": status,
                                      **state.get("failed_info", {})}) + "\n")
            inv_log.flush()
        inv_log.close()
        end = rec.mark("record_end", extra={"status": status})
        snap = torch.cuda.memory._snapshot()
        with (out / "memory_snapshot.pickle").open("wb") as f:
            pickle.dump(snap, f)
        torch.cuda.memory._record_memory_history(enabled=None)
        meta["status"] = status
        traces = snap.get("device_traces", [[]])
        meta["snapshot_trace_entries"] = len(traces[0])
        # [inugpu mod 7] truncation check: #alloc / #segment_alloc events in trace vs cumulative counter increase
        acts = {}
        for e in traces[0]:
            acts[e["action"]] = acts.get(e["action"], 0) + 1
        d_alloc = end["counters"]["alloc"] - start["counters"]["alloc"]
        d_seg = end["counters"]["segment_alloc"] - start["counters"]["segment_alloc"]
        meta["trace_check"] = {
            "n_devices_in_trace": len(traces), "action_counts": acts,
            "counter_delta_allocation_all_allocated": d_alloc, "counter_delta_allocation_all_freed":
                end["counters"]["free"] - start["counters"]["free"],
            "counter_delta_segment_all_allocated": d_seg, "counter_delta_segment_all_freed":
                end["counters"]["segment_free"] - start["counters"]["segment_free"],
            "alloc_events_equal_counter": acts.get("alloc", 0) == d_alloc,
            "segment_alloc_events_equal_counter": acts.get("segment_alloc", 0) == d_seg,
            "max_entries": a.max_entries, "entries_below_max": len(traces[0]) < a.max_entries,
            "first_event_action": traces[0][0]["action"] if traces[0] else None,
        }
        meta["truncated"] = not (meta["trace_check"]["alloc_events_equal_counter"] and meta["trace_check"]["entries_below_max"])
        (out / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str))
        say("trace_check", json.dumps(meta["trace_check"]))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
