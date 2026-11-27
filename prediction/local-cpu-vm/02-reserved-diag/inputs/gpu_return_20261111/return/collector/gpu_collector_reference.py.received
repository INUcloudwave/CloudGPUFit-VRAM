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
    a = ap.parse_args()
    manifest = json.loads(Path(a.manifest).read_text())
    case = next(c for c in manifest["cases"] if c["case_id"] == a.case)
    run = next(r for r in manifest["runs"] if r["run_id"] == a.run_id)
    out = Path(a.out) / a.case / a.run_id
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
        "case_id": a.case, "run_id": a.run_id, "run_group": run["group"], "steps": run["steps"],
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
        return {"input_ids_sha256": got}

    def finite_check():
        model, inputs = state["model"], state["inputs"]
        with torch.inference_mode():
            probe = model(**{n: v.to("cuda") for n, v in inputs.items()})
            finite = bool(torch.isfinite(probe.logits).all().item())
            del probe
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
        if new != n:
            raise RuntimeError("actual output length differs from the condition")
        return {"actual_new_tokens_per_sample": int(new), "output_ids_sha256": sha_tensor(host), "elapsed_ms": (t1 - t0) / 1e6}

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
    try:
        for phase, count in run["steps"]:
            step = phase.split("/")[-1]
            if step == "measure":
                torch.cuda.reset_peak_memory_stats()   # worker: before the measurement block (peak reset only)
            for i in range(count):
                rec.mark("phase_begin", phase, i)
                info = fns[step]()
                rec.mark("phase_end", phase, i, segments=(i == count - 1))
                inv_log.write(json.dumps({"phase": phase, "invocation": i, **info}) + "\n")
                inv_log.flush()
                say(phase, i, json.dumps(info))
    except torch.cuda.OutOfMemoryError:
        status = "gpu_oom"
        say(traceback.format_exc())
    except Exception:
        status = "error"
        say(traceback.format_exc())
    finally:
        rec.mark("record_end", extra={"status": status})
        snap = torch.cuda.memory._snapshot()
        with (out / "memory_snapshot.pickle").open("wb") as f:
            pickle.dump(snap, f)
        torch.cuda.memory._record_memory_history(enabled=None)
        meta["status"] = status
        meta["snapshot_trace_entries"] = len(snap.get("device_traces", [[]])[0])
        (out / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
