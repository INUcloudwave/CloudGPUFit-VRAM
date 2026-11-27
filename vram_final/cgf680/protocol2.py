"""Protocol v2 = INUGPU vram_worker.py (sha d06f0f15…) semantics, device-agnostic parts.

Differences from protocol v1 (all to match the GPU worker):
  * attention: transformers default (attn_implementation omitted)
  * generation: GenerationConfig.from_model_config(cfg) + update(greedy, min=max=N, ...) -> hub
    generation_config.json is NOT used; model.generation_config = gen; generate(**gi, generation_config=gen)
  * special ids replaced by 3: pad, eos, bos, decoder_start
  * every invocation copies the host inputs to the device (CPU: clone) and keeps them with the output
    until the invocation's record is taken, then both are deleted
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

SEED = 20261006
N_WARMUP = 3
LOADER = {"decoder_llm": "AutoModelForCausalLM", "text_encdec": "AutoModelForSeq2SeqLM",
          "vision_cls": "AutoModelForImageClassification"}


def sha_tensor(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def load(case, pin, dtype):
    import transformers
    cls = getattr(transformers, LOADER[case["kind"]])
    return cls.from_pretrained(case["model"], revision=pin["revision"], torch_dtype=dtype, low_cpu_mem_usage=True,
                               use_safetensors=pin["use_safetensors"], local_files_only=True)


def finish_load(model, dtype):
    model = model.to(device="cpu", dtype=dtype).eval()
    model.config.output_attentions = False
    model.config.output_hidden_states = False
    return model


def special_ids(cfg):
    out = []
    for name in ("pad_token_id", "eos_token_id", "bos_token_id", "decoder_start_token_id"):
        v = getattr(cfg, name, None)
        out += v if isinstance(v, list) else [v] if isinstance(v, int) else []
    return out


def host_inputs(case, cfg):
    g = torch.Generator().manual_seed(SEED)
    B = case["batch_size"]
    if case["kind"] == "vision_cls":
        r = case["resolution"]
        return {"pixel_values": torch.randn(B, 3, r, r, generator=g, dtype=torch.float32).to(torch.float16)}, None
    L = case["input_len"]
    ids = torch.randint(3, cfg.vocab_size, (B, L), generator=g, dtype=torch.int64)
    sp = special_ids(cfg)
    for tok in sp:
        ids[ids == tok] = 3
    return {"input_ids": ids, "attention_mask": torch.ones(B, L, dtype=torch.int64)}, sp


def check_valid(case, cfg):
    if case["kind"] == "decoder_llm":
        mp = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_positions", None)
        if mp is not None and case["input_len"] + case["new_tokens"] > mp:
            return f"input_len+new {case['input_len'] + case['new_tokens']} > max_position_embeddings {mp}"
    return None


def generation_config(model, case):
    from transformers import GenerationConfig
    cfg = model.config
    eos = cfg.eos_token_id[0] if isinstance(cfg.eos_token_id, list) else cfg.eos_token_id
    pad = cfg.pad_token_id if cfg.pad_token_id is not None else eos
    N = case["new_tokens"]
    gen = GenerationConfig.from_model_config(cfg)
    gen.update(do_sample=False, num_beams=1, use_cache=True, min_new_tokens=N, max_new_tokens=N, pad_token_id=pad,
               return_dict_in_generate=False, output_scores=False, output_attentions=False, output_hidden_states=False)
    model.generation_config = gen
    return gen


def invoke(model, case, inputs):
    """One invocation exactly like vram_worker.invoke (device copy of inputs inside). Returns (gi, o, new)."""
    gi = {k: v.clone() for k, v in inputs.items()}   # == v.to("cuda") on the GPU: a fresh device copy
    if case["kind"] == "vision_cls":
        o = model(**gi)
        return gi, o, None
    o = model.generate(**gi, generation_config=model.generation_config)
    new = o.shape[1] - (case["input_len"] if case["kind"] == "decoder_llm" else 1)
    return gi, o, int(new)


class InvalidInput(RuntimeError):
    pass


class LengthMismatch(RuntimeError):
    pass
