#!/usr/bin/env python3
"""CloudGPUFit FP16 VRAM worker. One process = one (condition, mode).

modes
  measure  : load once -> warmup x3 -> measure x5 (allocator cache kept). No performance mode, no pre-validation forward.
  validate : separate process; one inference with finiteness check of every step's logits + output length check.
  trace    : separate process; allocator history (alloc/free/segment events) for load -> warmup x3 -> measure x1,
             with counter boundaries, for xMem replay.

python -I vram_worker.py --models models.json --conditions conditions.json --condition ID --mode measure --out OUT.json
"""
import argparse
import gc
import gzip
import hashlib
import json
import os
import pickle
import platform
import statistics
import sys
import threading
import time
import traceback
from pathlib import Path

if os.environ.get("CUDA_MODULE_LOADING") is None:
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
MODULE_LOADING_ENV = os.environ.get("CUDA_MODULE_LOADING")

import torch  # noqa: E402

WORKER_VERSION = "cgf-fp16-vram-worker/1"
STAT_KEYS = ["allocated_bytes.all.current", "allocated_bytes.all.peak", "reserved_bytes.all.current",
             "reserved_bytes.all.peak", "active_bytes.all.peak", "requested_bytes.all.peak", "segment.all.current",
             "allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed",
             "num_alloc_retries", "num_ooms", "num_device_alloc", "num_device_free"]


def sha_tensor(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def module_loading_mode():
    try:
        import ctypes
        lib = ctypes.CDLL("libcuda.so.1")
        mode = ctypes.c_int(0)
        rc = lib.cuModuleGetLoadingMode(ctypes.byref(mode))
        return {"rc": rc, "mode": {1: "EAGER", 2: "LAZY"}.get(mode.value, mode.value)}
    except Exception as exc:
        return {"error": repr(exc)}


def stats():
    s = torch.cuda.memory_stats()
    return {k: s.get(k) for k in STAT_KEYS}


def classify(exc):
    msg = str(exc)
    if isinstance(exc, torch.cuda.OutOfMemoryError) or "CUDA out of memory" in msg:
        return "gpu_oom"
    if isinstance(exc, MemoryError) or "DefaultCPUAllocator: can't allocate memory" in msg:
        return "cpu_oom"
    if "not implemented for 'Half'" in msg or "\"Half\"" in msg:
        return "fp16_unsupported"
    if isinstance(exc, InvalidInput):
        return "invalid_input"
    if isinstance(exc, LengthMismatch):
        return "generation_length_mismatch"
    if isinstance(exc, NonFinite):
        return "nonfinite_output"
    return "compat_or_runtime_error"


class InvalidInput(RuntimeError):
    pass


class LengthMismatch(RuntimeError):
    pass


class NonFinite(RuntimeError):
    pass


class NvmlSampler:
    """Samples NVML per-process used memory of THIS pid (container must use --pid=host)."""

    def __init__(self, interval_s):
        import pynvml
        self.nv = pynvml
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.interval = interval_s
        self.pid = os.getpid()

    def _one(self):
        try:
            for p in self.nv.nvmlDeviceGetComputeRunningProcesses(self.h):
                if p.pid == self.pid and p.usedGpuMemory is not None:
                    return int(p.usedGpuMemory)
            return None
        except Exception as exc:  # noqa
            self.errors.append(repr(exc))
            return None

    def start(self):
        self.vals, self.errors, self.n = [], [], 0
        self.stop_ev = threading.Event()

        def loop():
            while True:
                v = self._one()
                self.n += 1
                if v is not None:
                    self.vals.append(v)
                if self.stop_ev.wait(self.interval):
                    break
        self.t = threading.Thread(target=loop, daemon=True)
        self.t.start()

    def stop(self):
        self.stop_ev.set()
        self.t.join()
        v = self._one()
        self.n += 1
        if v is not None:
            self.vals.append(v)
        return {"nvml_process_peak_bytes": max(self.vals) if self.vals else None, "nvml_samples": self.n,
                "nvml_samples_with_value": len(self.vals), "nvml_interval_s": self.interval,
                "nvml_errors": self.errors[:5]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--mode", choices=["measure", "validate", "trace"], default="measure")
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--measure", type=int, default=5)
    ap.add_argument("--nvml-interval", type=float, default=0.01)
    ap.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--attn", choices=["eager", "default"], default="eager")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print("refuse to overwrite", out, file=sys.stderr)
        return 3

    mcfg = json.loads(Path(a.models).read_text())
    cond = next(c for c in json.loads(Path(a.conditions).read_text())["conditions"] if c["condition_id"] == a.condition)
    model_id, kind, B = cond["model"], cond["workload_type"], cond["batch_size"]
    pin = mcfg["models"][model_id]
    seed = mcfg["common"]["seed"]
    n_warm, n_meas = (a.warmup, a.measure) if a.mode == "measure" else ((a.warmup, 1) if a.mode == "trace" else (0, 1))

    R = {"schema": WORKER_VERSION, "mode": a.mode, "condition": cond, "status": "running", "stage": "init",
         "worker_sha256": sha_file(__file__), "models_json_sha256": sha_file(a.models),
         "conditions_json_sha256": sha_file(a.conditions), "argv": sys.argv,
         "protocol": {"warmup": n_warm, "measure": n_meas, "allocator_cache_kept_between_iterations": True,
                      "peak_reset": "torch.cuda.reset_peak_memory_stats() right before each measure iteration (after start values)",
                      "end_values": "end_* recorded after synchronize with output tensor and GPU inputs still referenced; "
                                    "released_* after del output/inputs + synchronize"}}
    timings = {}
    t_main = time.perf_counter()

    def save():
        R["timings_s"] = timings
        R["worker_wall_s"] = time.perf_counter() - t_main
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(R, indent=1, default=str))
        tmp.replace(out)

    history = None
    try:
        torch.set_num_threads(a.threads)
        torch.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "need exactly one visible GPU"
        if a.mode == "trace":
            # only action/addr/size/stream/time are needed for replay; no stack frames (disk)
            torch.cuda.memory._record_memory_history(enabled="all", context=None, stacks="python", max_entries=50_000_000)
            history = {"enabled": "all", "context": None, "max_entries": 50_000_000, "boundaries": []}
        torch.cuda.init()
        torch.cuda.synchronize()
        import transformers
        from transformers import GenerationConfig
        props = torch.cuda.get_device_properties(0)
        env = {"gpu_name": props.name, "gpu_uuid": str(getattr(props, "uuid", None)), "cuda_total_memory": props.total_memory,
               "capability": list(torch.cuda.get_device_capability(0)), "torch": torch.__version__,
               "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
               "transformers": transformers.__version__, "python": platform.python_version(), "kernel": platform.release(),
               "cuda_module_loading_env": MODULE_LOADING_ENV, "cuda_module_loading_driver": module_loading_mode(),
               "allocator_backend": torch.cuda.get_allocator_backend(),
               "env": {k: os.environ.get(k) for k in ["PYTORCH_CUDA_ALLOC_CONF", "CUBLAS_WORKSPACE_CONFIG",
                                                     "CUBLASLT_WORKSPACE_SIZE", "PYTORCH_NO_CUDA_MEMORY_CACHING",
                                                     "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS"]},
               "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
               "cudnn_benchmark": torch.backends.cudnn.benchmark, "torch_threads": torch.get_num_threads(),
               "cpu_count": os.cpu_count(), "pid": os.getpid(),
               "image_id": os.environ.get("CGF_IMAGE_ID"), "gpu_label": os.environ.get("CGF_GPU_LABEL"),
               "instance_type": os.environ.get("CGF_INSTANCE_TYPE")}
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            env.update(driver=pynvml.nvmlSystemGetDriverVersion(), nvml_uuid=pynvml.nvmlDeviceGetUUID(h),
                       nvml_total=pynvml.nvmlDeviceGetMemoryInfo(h).total,
                       other_gpu_pids=[p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h) if p.pid != os.getpid()])
        except Exception as exc:
            env["nvml_error"] = repr(exc)
        R["environment"] = env
        if env.get("other_gpu_pids"):
            raise RuntimeError(f"other GPU processes present: {env['other_gpu_pids']}")
        R["start_state"] = stats()

        def boundary(name):
            if history is not None:
                torch.cuda.synchronize()
                history["boundaries"].append({"name": name, **stats()})

        # ---- model load
        R["stage"] = "model_load"
        boundary("model_load_begin")
        t0 = time.perf_counter()
        cls = getattr(transformers, pin["loader"])
        torch.cuda.reset_peak_memory_stats()
        # --attn eager: attn_implementation="eager" (pilot v1). --attn default: argument omitted -> transformers default
        attn_kw = {"attn_implementation": "eager"} if a.attn == "eager" else {}
        model = cls.from_pretrained(model_id, revision=pin["revision"], torch_dtype=torch.float16, low_cpu_mem_usage=True,
                                    device_map={"": "cuda:0"}, use_safetensors=pin["use_safetensors"],
                                    local_files_only=True, **attn_kw)
        model = model.to(device="cuda", dtype=torch.float16).eval()
        model.config.output_attentions = False
        model.config.output_hidden_states = False
        torch.cuda.synchronize()
        timings["model_load"] = time.perf_counter() - t0
        boundary("model_load_end")
        R["after_load"] = {**stats(), "max_allocated": torch.cuda.max_memory_allocated(),
                           "max_reserved": torch.cuda.max_memory_reserved()}
        cfg = model.config
        R["model"] = {"id": model_id, "revision": pin["revision"], "loader": pin["loader"],
                      "use_safetensors": pin["use_safetensors"], "class": type(model).__name__,
                      "attn_implementation": getattr(cfg, "_attn_implementation", None),
                      "attn_requested": a.attn,
                      "sdpa_backends_enabled": {"flash": torch.backends.cuda.flash_sdp_enabled(),
                                                "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                                                "math": torch.backends.cuda.math_sdp_enabled(),
                                                "cudnn": torch.backends.cuda.cudnn_sdp_enabled()},
                      "param_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
                      "param_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
                      "buffer_bytes": sum(b.numel() * b.element_size() for b in model.buffers())}

        # ---- inputs on host
        R["stage"] = "inputs"
        g = torch.Generator().manual_seed(seed)
        if kind == "vision_cls":
            r = cond["resolution"]
            px = torch.randn(B, 3, r, r, generator=g, dtype=torch.float32).to(torch.float16)
            if getattr(cfg, "num_channels", 3) != 3:
                raise InvalidInput(f"num_channels {cfg.num_channels}")
            inputs = {"pixel_values": px}
        else:
            L, N = cond["input_len"], cond["max_new_tokens"]
            ids = torch.randint(3, cfg.vocab_size, (B, L), generator=g, dtype=torch.int64)
            special = []
            for name in ("pad_token_id", "eos_token_id", "bos_token_id", "decoder_start_token_id"):
                v = getattr(cfg, name, None)
                special += v if isinstance(v, list) else [v] if isinstance(v, int) else []
            for tok in special:
                ids[ids == tok] = 3
            inputs = {"input_ids": ids, "attention_mask": torch.ones(B, L, dtype=torch.int64)}
            if kind == "decoder_llm":
                maxpos = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_positions", None)
                R["max_position_embeddings"] = maxpos
                if maxpos is not None and L + N > maxpos:
                    raise InvalidInput(f"input_len+new {L + N} > max_position_embeddings {maxpos}")
            eos = cfg.eos_token_id[0] if isinstance(cfg.eos_token_id, list) else cfg.eos_token_id
            pad = cfg.pad_token_id if cfg.pad_token_id is not None else eos
            gen = GenerationConfig.from_model_config(cfg)
            gen.update(do_sample=False, num_beams=1, use_cache=True, min_new_tokens=N, max_new_tokens=N,
                       pad_token_id=pad, return_dict_in_generate=False, output_scores=False,
                       output_attentions=False, output_hidden_states=False)
            model.generation_config = gen   # replaces hub generation_config.json (no sampling/penalty knobs)
            R["generation_config"] = gen.to_dict()
        R["inputs"] = {k: {"shape": list(v.shape), "dtype": str(v.dtype), "sha256": sha_tensor(v)} for k, v in inputs.items()}
        R["special_tokens_replaced_by_3"] = None if kind == "vision_cls" else special

        finite_flags = []
        if a.mode == "validate":
            # raw model logits of every forward (before logits processors such as min_new_tokens' EOS=-inf mask)
            def finite_hook(module, args_, output):
                finite_flags.append(torch.isfinite(output.logits).all())
            model.register_forward_hook(finite_hook)

        def invoke():
            gi = {k: v.to("cuda") for k, v in inputs.items()}
            if kind == "vision_cls":
                o = model(**gi)
                if tuple(o.logits.shape) != (B, cfg.num_labels):
                    raise InvalidInput(f"logits shape {tuple(o.logits.shape)}")
                return gi, o, None
            o = model.generate(**gi, generation_config=model.generation_config)
            new = o.shape[1] - (cond["input_len"] if kind == "decoder_llm" else 1)
            if o.shape[0] != B or new != cond["max_new_tokens"]:
                raise LengthMismatch(f"output shape {tuple(o.shape)} -> new tokens {new}")
            return gi, o, int(new)

        # ---- warmup
        R["warmup"] = []
        for i in range(n_warm):
            R["stage"] = f"warmup_{i}"
            boundary(f"warmup_{i}_begin")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                gi, o, new = invoke()
            torch.cuda.synchronize()
            el = time.perf_counter() - t0
            end = stats()
            del gi, o
            torch.cuda.synchronize()
            boundary(f"warmup_{i}_end")
            R["warmup"].append({"i": i, "elapsed_s": el, "new_tokens": new, "end_with_output": end,
                                "released": {"allocated": torch.cuda.memory_allocated(),
                                             "reserved": torch.cuda.memory_reserved()}})

        # ---- measure
        sampler = NvmlSampler(a.nvml_interval) if a.mode == "measure" else None
        R["measures"] = []
        for i in range(n_meas):
            R["stage"] = f"measure_{i}"
            torch.cuda.synchronize()
            st = {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved()}
            boundary(f"measure_{i}_begin")
            torch.cuda.reset_peak_memory_stats()
            if sampler:
                sampler.start()
            t0 = time.perf_counter()
            with torch.inference_mode():
                gi, o, new = invoke()
            torch.cuda.synchronize()
            el = time.perf_counter() - t0
            row = {"i": i, "ok": True, "new_tokens": new, "elapsed_s": el, "start": st,
                   "peak": {"allocated": torch.cuda.max_memory_allocated(), "reserved": torch.cuda.max_memory_reserved()},
                   "end_with_output": {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved()},
                   "stats_end": stats()}
            if sampler:
                row.update(sampler.stop())
            host = o.cpu() if torch.is_tensor(o) else o.logits.cpu()
            row["output_sha256"] = sha_tensor(host)
            del gi, o
            torch.cuda.synchronize()
            boundary(f"measure_{i}_end")
            row["released"] = {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved()}
            R["measures"].append(row)
        if a.mode == "validate":
            fl = [bool(x.item()) for x in finite_flags]
            R["validation"] = {"steps_checked": len(fl), "all_finite": all(fl) if fl else None,
                               "first_nonfinite_step": fl.index(False) if False in fl else None}
            if fl and not all(fl):
                R["status"] = "nonfinite_output"
        if a.mode == "measure":
            pk = [m["peak"]["reserved"] for m in R["measures"]]
            pa = [m["peak"]["allocated"] for m in R["measures"]]
            nv = [m["nvml_process_peak_bytes"] for m in R["measures"] if m.get("nvml_process_peak_bytes") is not None]
            R["summary"] = {"first_measure_peak_reserved": pk[0], "first_measure_peak_allocated": pa[0],
                            "first_measure_start_reserved": R["measures"][0]["start"]["reserved"],
                            "first_measure_start_allocated": R["measures"][0]["start"]["allocated"],
                            "median_peak_reserved": statistics.median(pk), "max_peak_reserved": max(pk),
                            "median_peak_allocated": statistics.median(pa), "max_peak_allocated": max(pa),
                            "max_nvml_process_peak": max(nv) if nv else None,
                            "median_nvml_process_peak": statistics.median(nv) if nv else None}
        if R["status"] == "running":
            R["status"] = "ok"
        R["stage"] = "complete"
    except BaseException as exc:  # noqa: BLE001 - record and classify everything
        R["status"] = classify(exc)
        R["failed_stage"] = R.get("stage")
        R["error"] = f"{type(exc).__name__}: {str(exc)[:2000]}"
        R["traceback"] = traceback.format_exc()[-6000:]
        try:
            R["stats_at_failure"] = stats()
        except Exception:
            pass
    finally:
        if history is not None:
            try:
                torch.cuda.synchronize()
                snap = torch.cuda.memory._snapshot()
                tr = snap["device_traces"][0]
                acts = {}
                for e in tr:
                    acts[e["action"]] = acts.get(e["action"], 0) + 1
                s_end = stats()
                history.update(action_counts=acts, end=s_end,
                               alloc_events_equal_counter=acts.get("alloc", 0) == s_end["allocation.all.allocated"],
                               start_empty=R.get("start_state", {}).get("reserved_bytes.all.current") == 0)
                p = out.with_name(out.stem + ".snapshot.pickle.gz")
                with gzip.open(p, "wb", compresslevel=6) as f:
                    pickle.dump(snap, f)
                history["snapshot_file"] = p.name
                torch.cuda.memory._record_memory_history(enabled=None)
            except Exception as exc:
                history["error"] = repr(exc)
            R["history"] = history
        save()
    return 0 if R["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
