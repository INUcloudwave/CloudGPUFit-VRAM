"""GPU-dependent constants for reserved prediction, from information known BEFORE running on the GPU:
the GPU model name (-> architecture / compute capability, public NVIDIA specs) and the runtime (PyTorch version,
environment variables). No GPU measurement is needed for these values.

cuBLAS workspace (one block per handle/stream, allocated through the caching allocator at the first cuBLAS call and
kept): PyTorch CublasHandlePool.cpp, parseChosenWorkspaceSize()
  * CUBLAS_WORKSPACE_CONFIG=":SIZE_KiB:COUNT[:SIZE_KiB:COUNT...]" overrides (sum of SIZE*COUNT KiB)
  * default: compute capability 9.0 (Hopper) -> 4096 KiB x 8 = 32 MiB; otherwise 4096 KiB x 2 + 16 KiB x 8 = 8,519,680 B
Checked against measurements: RTX 3090 (8.6) 8,519,680 B with torch 2.6.0 (GPU traces 12/12) and torch 2.7.1
(INUGPU cu128 run: measure-start allocated byte-equal to 2.6). Hopper / Blackwell values are NOT yet checked against a
measurement (see VERIFIED).
"""
from __future__ import annotations

import re

# GPU model name -> (architecture, compute capability). Public NVIDIA specifications.
GPU_ARCH = {
    "NVIDIA T4": ("Turing", (7, 5)),
    "Tesla T4": ("Turing", (7, 5)),
    "NVIDIA A10G": ("Ampere", (8, 6)),
    "NVIDIA GeForce RTX 3090": ("Ampere", (8, 6)),
    "NVIDIA L4": ("Ada", (8, 9)),
    "NVIDIA L40S": ("Ada", (8, 9)),
    "NVIDIA H100": ("Hopper", (9, 0)),
    "NVIDIA RTX PRO 4500 Blackwell Server Edition": ("Blackwell", (12, 0)),
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": ("Blackwell", (12, 0)),
}

# (torch major.minor, architecture) combinations whose workspace size was confirmed by a GPU measurement.
# 2026-11-18: AWS T4 / A10G / L4 / L40S (torch 2.6.0) and RTX PRO 4500 Blackwell (torch 2.7.1): measure-start minus
# after-load allocated equals the RTX 3090 value on every decoder condition (190-192 per GPU), i.e. 8,519,680 B
# (results/reserved_v3/aws_gpu_check).
VERIFIED = {("2.6", "Ampere"), ("2.7", "Ampere"), ("2.6", "Turing"), ("2.6", "Ada"), ("2.7", "Blackwell")}

DEFAULT_SMALL = 4096 * 1024 * 2 + 16 * 1024 * 8      # 8,519,680
DEFAULT_SM90 = 4096 * 1024 * 8                       # 33,554,432


def arch_of(gpu_name):
    """Architecture and compute capability from the GPU model name (exact name first, then longest contained key)."""
    if gpu_name in GPU_ARCH:
        return GPU_ARCH[gpu_name]
    hits = [k for k in GPU_ARCH if k.lower() in gpu_name.lower()]
    if not hits:
        raise KeyError(f"unknown GPU model name: {gpu_name!r} (add it to cgf680.gpu_rules.GPU_ARCH)")
    return GPU_ARCH[max(hits, key=len)]


def parse_workspace_config(cfg):
    """':4096:8' / ':4096:2:16:8' -> bytes (sum of SIZE_KiB * COUNT)."""
    nums = [int(x) for x in re.findall(r"\d+", cfg)]
    if not nums or len(nums) % 2:
        raise ValueError(f"bad CUBLAS_WORKSPACE_CONFIG: {cfg!r}")
    return sum(nums[i] * 1024 * nums[i + 1] for i in range(0, len(nums), 2))


def cublas_workspace(gpu_name=None, compute_capability=None, torch_version="2.6.0", env_config=None):
    """-> dict(bytes, rule, arch, compute_capability, verified)."""
    arch = None
    if compute_capability is None:
        arch, compute_capability = arch_of(gpu_name)
    elif gpu_name:
        arch = arch_of(gpu_name)[0]
    cc = tuple(compute_capability)
    tv = ".".join(str(torch_version).split("+")[0].split(".")[:2])
    if env_config:
        return {"bytes": parse_workspace_config(env_config), "rule": f"CUBLAS_WORKSPACE_CONFIG={env_config}",
                "arch": arch, "compute_capability": list(cc), "verified": True}
    sm90 = cc == (9, 0)
    return {"bytes": DEFAULT_SM90 if sm90 else DEFAULT_SMALL,
            "rule": f"PyTorch {tv} default: " + ("cc 9.0 -> 4096 KiB x 8" if sm90 else "4096 KiB x 2 + 16 KiB x 8"),
            "arch": arch, "compute_capability": list(cc), "verified": (tv, arch) in VERIFIED}
