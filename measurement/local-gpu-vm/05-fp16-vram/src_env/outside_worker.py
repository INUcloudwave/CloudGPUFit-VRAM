#!/usr/bin/env python3
"""Run the unchanged /src/vram_worker.py and record PyTorch + NVML memory at three fixed points (env comparison, 2026-11-17).

  python -I outside_worker.py <vram_worker args...>      (measure mode)

Points (all after torch.cuda.synchronize(); torch allocated/reserved and NVML per-process used memory read back to back):
  cuda_init      right after the worker's torch.cuda.init() (+ synchronize, which the worker also does next)
  after_load     first worker call of torch.cuda.max_memory_allocated() = right after model load + synchronize
  after_warmup   second worker call of torch.cuda.reset_peak_memory_stats() = start of measure_0, i.e. after the
                 3 warmups and their release (del output + synchronize); recorded BEFORE the peak reset
outside = nvml_process_used - torch reserved (same point). A second NVML read 200 ms later is kept for stability.
Only calls made directly from vram_worker.py count (calls from inside transformers/accelerate are ignored).
Result: <out>.outside.json next to the worker result. vram_worker.py itself is not modified (its sha is unchanged).
None of these hooks allocate GPU memory; the 200 ms waits happen outside the timed measure iterations.
"""
import json
import os
import runpy
import sys
import time
from pathlib import Path

if os.environ.get("CUDA_MODULE_LOADING") is None:
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
import torch  # noqa: E402

WORKER = "/src/vram_worker.py"
info = {"schema": "cgf-outside-points/1", "points": {}, "hook_calls": [], "errors": []}
_nv = {}


def nvml_used():
    try:
        if not _nv:
            import pynvml
            pynvml.nvmlInit()
            _nv.update(m=pynvml, h=pynvml.nvmlDeviceGetHandleByIndex(0))
        for p in _nv["m"].nvmlDeviceGetComputeRunningProcesses(_nv["h"]):
            if p.pid == os.getpid():
                return int(p.usedGpuMemory) if p.usedGpuMemory is not None else None
        return None
    except Exception as exc:  # noqa
        info["errors"].append(repr(exc))
        return None


def point(name, lineno):
    if name in info["points"]:
        return
    torch.cuda.synchronize()
    alloc, res = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    used = nvml_used()
    s = torch.cuda.memory_stats()
    rec = {"worker_line": lineno, "t_unix": time.time(), "allocated": alloc, "reserved": res,
           "nvml_process_used": used, "outside": (used - res) if used is not None else None,
           "segments": s.get("segment.all.current"), "num_alloc_retries": s.get("num_alloc_retries"),
           "num_ooms": s.get("num_ooms"), "num_device_alloc": s.get("num_device_alloc")}
    time.sleep(0.2)
    rec["nvml_process_used_after_200ms"] = nvml_used()
    info["points"][name] = rec


def from_worker():
    f = sys._getframe(2)
    return f.f_code.co_filename == WORKER, f.f_lineno


_init = torch.cuda.init
_maxalloc = torch.cuda.max_memory_allocated
_reset = torch.cuda.reset_peak_memory_stats
_n_reset = [0]


def init_hook():
    _init()
    ok, ln = from_worker()
    if ok:
        point("cuda_init", ln)


def maxalloc_hook(*a, **k):
    ok, ln = from_worker()
    if ok:
        point("after_load", ln)
    return _maxalloc(*a, **k)


def reset_hook(*a, **k):
    ok, ln = from_worker()
    if ok:
        _n_reset[0] += 1
        info["hook_calls"].append({"fn": "reset_peak_memory_stats", "n": _n_reset[0], "worker_line": ln})
        if _n_reset[0] == 2:
            point("after_warmup", ln)
    return _reset(*a, **k)


torch.cuda.init = init_hook
torch.cuda.max_memory_allocated = maxalloc_hook
torch.cuda.reset_peak_memory_stats = reset_hook

argv = sys.argv[1:]
out = Path(argv[argv.index("--out") + 1])
info["env"] = {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
               "CUDA_MODULE_LOADING": os.environ.get("CUDA_MODULE_LOADING"),
               "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
               "env_name": os.environ.get("CGF_ENV_NAME")}
try:
    from importlib import metadata
    info["env"]["pip_cuda_packages"] = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()
                                              if d.metadata["Name"] and (d.metadata["Name"].lower().startswith("nvidia-")
                                                                         or d.metadata["Name"].lower() in ("torch", "triton", "torchvision", "torchaudio")))
except Exception as exc:  # noqa
    info["errors"].append(repr(exc))
sys.argv = [WORKER] + argv
try:
    runpy.run_path(WORKER, run_name="__main__")
finally:
    try:
        import pynvml
        info["env"]["driver"] = pynvml.nvmlSystemGetDriverVersion() if _nv else None
    except Exception as exc:  # noqa
        info["errors"].append(repr(exc))
    out.with_suffix(".outside.json").write_text(json.dumps(info, indent=1))
