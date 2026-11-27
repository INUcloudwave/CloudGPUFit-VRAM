"""Synthesize the GPU allocator request sequence of `from_pretrained(..., torch_dtype=fp16, device_map=cuda,
low_cpu_mem_usage=True)` + `.to(cuda, fp16)` (transformers 4.52.4), without a GPU.

Rules (derived from INUGPU RTX 3090 load-phase traces, validated by tests/probes/validate_load_emu.py):
  1. caching_allocator_warmup: alloc W then free, W = sum over model.state_dict() entries (tied keys counted
     separately, non-persistent buffers excluded) of numel x load-dtype size (fp16; fp32 for
     _keep_in_fp32_modules; integer dtypes keep their size).
  2. checkpoint tensors in checkpoint order (safetensors: sorted keys; .bin: file order), keys the model
     does not expect are skipped. Per tensor: if checkpoint dtype == load dtype -> alloc(load bytes);
     else alloc(ckpt bytes), alloc(load bytes), free(ckpt bytes). A checkpoint key whose parameter is tied
     to an already-loaded one is allocated and freed right away (tie_weights replaces it).
  3. buffers not in the checkpoint (non-persistent) materialized in module order: alloc(nbytes in final dtype).
  4. protocol `.to(fp16)`: every floating tensor that is not fp16 yet (keep-fp32 params, fp32 rotary buffers)
     is cast in Module._apply order (children first): alloc(fp16), free(original).
  Checkpoints without the base_model_prefix (GPT-2) are loaded into the base model: W/keys use that submodule.
Returned as a list of (+bytes | -bytes) with stable ids so it can be replayed before the traced requests;
the model's resident storages at the end are exactly the live blocks.
"""
from __future__ import annotations

import glob
import json
import math
import struct

import torch

DT_SIZE = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "BOOL": 1, "U8": 1, "F64": 8}
TORCH_DT = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16", torch.int64: "I64",
            torch.int32: "I32", torch.bool: "BOOL", torch.uint8: "U8", torch.float64: "F64"}


def checkpoint_entries(snapshot_dir, use_safetensors):
    if use_safetensors:
        files = sorted(glob.glob(f"{snapshot_dir}/*.safetensors"))
        out = []
        for f in files:
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                h = json.loads(fh.read(n))
            h.pop("__metadata__", None)
            out += [(k, h[k]["dtype"], h[k]["shape"]) for k in sorted(h)]
        return out
    sd = torch.load(glob.glob(f"{snapshot_dir}/pytorch_model.bin")[0], map_location="meta", weights_only=True)
    return [(k, TORCH_DT[v.dtype], list(v.shape)) for k, v in sd.items()]


def _post_order_tensors(module, prefix=""):
    """Order in which Module._apply (used by .to()) visits tensors: children first, then own params, buffers."""
    for name, child in module.named_children():
        yield from _post_order_tensors(child, f"{prefix}{name}.")
    for name, p in module._parameters.items():
        if p is not None:
            yield f"{prefix}{name}", p
    for name, b in module._buffers.items():
        if b is not None:
            yield f"{prefix}{name}", b


def synthesize(model_as_loaded, snapshot_dir, use_safetensors, target_dtype=torch.float16):
    """model_as_loaded: from_pretrained(torch_dtype=fp16) result BEFORE the protocol .to(fp16) (any device);
    used for structure and per-tensor dtypes only."""
    model = model_as_loaded
    ckpt = checkpoint_entries(snapshot_dir, use_safetensors)
    prefix = getattr(model, "base_model_prefix", "") or ""
    target, tprefix = model, ""
    if prefix and hasattr(model, prefix) and not any(k.startswith(prefix + ".") for k, _, _ in ckpt):
        target, tprefix = getattr(model, prefix), prefix + "."   # checkpoint of the base model -> load into it
    keep32 = set(getattr(model, "_keep_in_fp32_modules", None) or [])

    def is_keep32(name):
        return any(f".{m}." in f".{name}." for m in keep32)

    sd = target.state_dict(keep_vars=True)

    def load_dt(name, t):
        if not t.is_floating_point():
            return TORCH_DT[t.dtype]
        return "F32" if is_keep32(tprefix + name) else "F16"

    seq = []
    W = sum(t.numel() * DT_SIZE[load_dt(k, t)] for k, t in sd.items())
    seq += [("alloc", "warmup", W), ("free", "warmup", W)]
    loaded_storages = set()
    stats = {"warmup_bytes": W, "ckpt_skipped_unexpected": 0, "tied_dup_loaded_then_freed": 0,
             "keep_fp32": sorted(keep32), "loaded_into": tprefix or "<model>", "post_load_casts": 0}
    labels = {}
    for k, dt, shape in ckpt:
        if k not in sd:
            stats["ckpt_skipped_unexpected"] += 1
            continue
        t = sd[k]
        n = math.prod(shape)
        ld = load_dt(k, t)
        key = t.untyped_storage().data_ptr()
        full = tprefix + k
        if dt == ld:
            seq.append(("alloc", full, n * DT_SIZE[ld]))
        else:
            seq += [("alloc", full + "#ckpt", n * DT_SIZE[dt]), ("alloc", full, n * DT_SIZE[ld]),
                    ("free", full + "#ckpt", n * DT_SIZE[dt])]
        if key in loaded_storages:
            seq.append(("free", full, n * DT_SIZE[ld]))
            stats["tied_dup_loaded_then_freed"] += 1
        else:
            labels[key] = (full, n * DT_SIZE[ld])
        loaded_storages.add(key)
    full_sd = set(model.state_dict().keys())
    for name, b in model.named_buffers():
        if name in full_sd:
            continue
        seq.append(("alloc", name, b.numel() * b.element_size()))
        labels[b.untyped_storage().data_ptr()] = (name, b.numel() * b.element_size())
    seen = set()
    for name, t in _post_order_tensors(model):   # protocol .to(fp16): cast every floating tensor not yet fp16
        key = t.untyped_storage().data_ptr()
        if key in seen:
            continue
        seen.add(key)
        if t.is_floating_point() and t.dtype != target_dtype and key in labels:
            lab, nbytes = labels[key]
            seq += [("alloc", lab + "#cast", t.numel() * torch.tensor([], dtype=target_dtype).element_size()), ("free", lab, nbytes)]
            stats["post_load_casts"] += 1
    return seq, stats


def to_events(seq):
    """-> list of (action 1/0, size, alloc_id), ids matched by label (frees of a label free its alloc)."""
    ids, out, nxt = {}, [], 0
    for act, label, size in seq:
        if act == "alloc":
            ids[label] = nxt
            out.append((1, int(size), nxt))
            nxt += 1
        else:
            out.append((0, int(size), ids.pop(label)))
    return out, len(ids)


def signed(seq):
    return [s if a == "alloc" else -s for a, _, s in seq]
