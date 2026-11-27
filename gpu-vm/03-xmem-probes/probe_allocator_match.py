import gc
import os
import torch

from perf_estimator.allocator.allocator import (
    CachingAllocator,
    Device,
    Stream,
)

KiB = 1024
MiB = 1024 * 1024

SCENARIOS = {
    "small_pool": [
        ("alloc", "a", 100 * KiB),
        ("alloc", "b", 900 * KiB),
        ("free",  "a", None),
        ("alloc", "c", 200 * KiB),
    ],

    "large_20mb": [
        ("alloc", "a", 2 * MiB),
        ("alloc", "b", 6 * MiB),
        ("alloc", "c", 7 * MiB),
    ],

    "large_over_10mb": [
        ("alloc", "a", 11 * MiB),
        ("alloc", "b", 13 * MiB),
    ],

    "fragmentation": [
        ("alloc", "a", 6 * MiB),
        ("alloc", "b", 6 * MiB),
        ("alloc", "c", 6 * MiB),
        ("free",  "b", None),
        ("alloc", "d", 7 * MiB),
    ],
}


def real_stats(base_alloc, base_res):
    torch.cuda.synchronize()
    return (
        torch.cuda.memory_allocated() - base_alloc,
        torch.cuda.memory_reserved() - base_res,
    )


def sim_stats(sim):
    return (
        sim._trace.max_usage_changes[-1],
        sim._trace.max_segment_changes[-1],
    )


def run_scenario(name, ops):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    base_alloc = torch.cuda.memory_allocated()
    base_res = torch.cuda.memory_reserved()

    sim = CachingAllocator(
        allowed_memory_maximum=24 * 1024**3
    )

    device = Device(index=0)
    stream = Stream(index=0)

    real_blocks = {}
    sim_blocks = {}

    print()
    print("=" * 90)
    print("SCENARIO:", name)
    print("=" * 90)

    for step, (action, key, size) in enumerate(ops, 1):

        if action == "alloc":
            real_blocks[key] = torch.empty(
                size,
                dtype=torch.uint8,
                device="cuda:0",
            )

            sim_blocks[key] = sim.malloc(
                device,
                stream,
                size,
            )

            desc = f"ALLOC {key} {size} B"

        elif action == "free":
            del real_blocks[key]
            gc.collect()

            sim.free(sim_blocks.pop(key))

            desc = f"FREE  {key}"

        else:
            raise ValueError(action)

        torch.cuda.synchronize()

        r_alloc, r_res = real_stats(base_alloc, base_res)
        s_alloc, s_res = sim_stats(sim)

        print(
            f"{step:02d} {desc:<24} | "
            f"REAL alloc={r_alloc:>10} reserved={r_res:>10} | "
            f"SIM alloc={s_alloc:>10} reserved={s_res:>10} | "
            f"RES_DIFF={s_res-r_res:+d}"
        )

    real_blocks.clear()
    sim_blocks.clear()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("PYTORCH_CUDA_ALLOC_CONF:",
      os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))

try:
    print(
        "allocator backend:",
        torch.cuda.memory.get_allocator_backend()
    )
except Exception as e:
    print("allocator backend: unavailable", e)

for name, ops in SCENARIOS.items():
    run_scenario(name, ops)
