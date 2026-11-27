#!/usr/bin/env python3
"""Run the unchanged /src/vram_worker.py with a hard allocator capacity (R_min validation on GPU).

  CGF_MEM_CAP_BYTES=<bytes> python -I cap_worker.py <vram_worker args...>

torch.cuda.set_per_process_memory_fraction(cap / total_memory) is applied right after the worker's own
torch.cuda.init(). The caching allocator then treats `reserved + request > cap` exactly like a cudaMalloc failure:
free cached blocks -> retry (num_alloc_retries) -> OutOfMemoryError (status gpu_oom). The cap and the applied
fraction are written to <out>.cap.json next to the worker result. vram_worker.py itself is not modified."""
import json
import os
import runpy
import sys
from pathlib import Path

cap = int(os.environ["CGF_MEM_CAP_BYTES"])
if os.environ.get("CUDA_MODULE_LOADING") is None:
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
import torch  # noqa: E402

_orig_init = torch.cuda.init
info = {"cap_bytes": cap}


def init_with_cap():
    _orig_init()
    total = torch.cuda.get_device_properties(0).total_memory
    frac = cap / total
    if not 0 < frac <= 1:
        raise SystemExit(f"cap {cap} outside (0, total={total}]")
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    info.update(total_memory=total, fraction=frac, allowed_bytes_expected=int(frac * total))


torch.cuda.init = init_with_cap
worker = "/src/vram_worker.py"
argv = sys.argv[1:]
out = Path(argv[argv.index("--out") + 1])
sys.argv = [worker] + argv
try:
    runpy.run_path(worker, run_name="__main__")
finally:
    out.with_suffix(".cap.json").write_text(json.dumps(info, indent=1))
