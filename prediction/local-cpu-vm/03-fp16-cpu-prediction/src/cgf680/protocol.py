"""Device-agnostic common protocol: model load, input generation, one inference invocation.

Both the GPU runner (device="cuda") and the CPU tracer (device="cpu") must call these functions so
that weights, inputs, and generate arguments are identical. Anything the GPU runner does differently
must be recorded as a protocol deviation.

Protocol (per case, fresh process):
  load pretrained FP16 (pinned revision, eager attention, eval)  ->  inputs generated on host
  (seed 20261006) and moved to the device ONCE  ->  warmup x3  ->  measure x1  (same inputs, output
  released between invocations; no validation forward / performance mode before measurement).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

SEED = 20261006
N_WARMUP = 3
N_MEASURE = 1
DTYPES = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
MODEL_CLASS = {"decoder_llm": "AutoModelForCausalLM", "text_encdec": "AutoModelForSeq2SeqLM",
               "vision_cls": "AutoModelForImageClassification"}


def tensor_sha256(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def load_pins(path):
    return json.loads(Path(path).read_text())


def load_model(case, pin, device):
    import transformers
    cls = getattr(transformers, MODEL_CLASS[case["kind"]])
    dtype = DTYPES[case["dtype"]]
    kw = dict(revision=pin["revision"], torch_dtype=dtype, low_cpu_mem_usage=True,
              use_safetensors=pin["safetensors"], local_files_only=True)
    attn = "eager"
    try:
        model = cls.from_pretrained(case["model"], attn_implementation="eager", **kw)
    except (ValueError, TypeError) as exc:  # models without an attention implementation switch
        if "attn_implementation" not in str(exc) and "attention" not in str(exc).lower():
            raise
        model = cls.from_pretrained(case["model"], **kw)
        attn = f"default (eager rejected: {type(exc).__name__})"
    model = model.to(device=device, dtype=dtype).eval()
    model.config.output_attentions = False
    model.config.output_hidden_states = False
    return model, attn


def _special_ids(config):
    ids = set()
    for name in ("pad_token_id", "eos_token_id", "bos_token_id"):
        v = getattr(config, name, None)
        for x in (v if isinstance(v, (list, tuple)) else [v]):
            if isinstance(x, int):
                ids.add(x)
    return sorted(ids)


def make_host_inputs(case, config):
    """Host (CPU) inputs. Text: exactly input_len random ids in [3, vocab), special ids -> 3, no BOS
    prepended, no truncation, int64 ids + int64 all-ones mask. Vision: float32 N(0,1) generated on host
    then cast to the case dtype, shape [B,3,R,R] (no image processor, so no resize to 224)."""
    g = torch.Generator().manual_seed(SEED)
    b = case["batch_size"]
    if case["kind"] == "vision_cls":
        r = case["resolution"]
        x = torch.randn(b, config.num_channels, r, r, generator=g, dtype=torch.float32).to(DTYPES[case["dtype"]])
        return {"pixel_values": x}
    length = case["input_len"]
    ids = torch.randint(3, config.vocab_size, (b, length), generator=g, dtype=torch.long)
    for tok in _special_ids(config):
        ids[ids == tok] = 3
    return {"input_ids": ids, "attention_mask": torch.ones(b, length, dtype=torch.long)}


def input_record(inputs):
    return {k: {"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", ""), "sha256": tensor_sha256(v)}
            for k, v in inputs.items()}


def validate_inputs(case, inputs):
    b = case["batch_size"]
    if case["kind"] == "vision_cls":
        x = inputs["pixel_values"]
        exp = [b, 3, case["resolution"], case["resolution"]]
        if list(x.shape) != exp or x.dtype != DTYPES[case["dtype"]]:
            raise InputMismatch(f"pixel_values {list(x.shape)} {x.dtype} != {exp} {case['dtype']}")
    else:
        for k in ("input_ids", "attention_mask"):
            if list(inputs[k].shape) != [b, case["input_len"]] or inputs[k].dtype != torch.long:
                raise InputMismatch(f"{k} {list(inputs[k].shape)} {inputs[k].dtype}")


def check_shape_supported(case, config):
    if case["kind"] == "decoder_llm":
        limit = getattr(config, "max_position_embeddings", None) or getattr(config, "n_positions", None)
        if limit is not None and case["input_len"] + case["new_tokens"] > limit:
            return f"input_len+new_tokens {case['input_len'] + case['new_tokens']} > max positions {limit}"
    if getattr(config, "model_type", "") == "vit" and case["resolution"] != config.image_size:
        return f"vit resolution {case['resolution']} != {config.image_size}"
    return None


def generate_kwargs(case, config):
    n = case["new_tokens"]
    pad = config.pad_token_id if config.pad_token_id is not None else (
        config.eos_token_id if isinstance(config.eos_token_id, int) else config.eos_token_id[0])
    return dict(do_sample=False, num_beams=1, use_cache=True, min_new_tokens=n, max_new_tokens=n, pad_token_id=pad,
                return_dict_in_generate=False, output_scores=False, output_attentions=False,
                output_hidden_states=False)


def invoke(model, case, inputs, gen_kwargs):
    """One inference invocation. Returns (output_tensor, actual_new_tokens_per_sample)."""
    with torch.inference_mode():
        if case["kind"] == "vision_cls":
            out = model(**inputs).logits
            return out, 0
        out = model.generate(**inputs, **gen_kwargs)
        prefix = case["input_len"] if case["kind"] == "decoder_llm" else 1  # T5: decoder_start excluded
        return out, int(out.shape[1] - prefix)


def resolved_generation_config(model, gen_kwargs):
    try:
        cfg = model.generation_config.to_dict()
    except Exception:
        return None
    keep = ("do_sample", "num_beams", "repetition_penalty", "temperature", "top_p", "top_k", "no_repeat_ngram_size",
            "bad_words_ids", "forced_bos_token_id", "forced_eos_token_id", "decoder_start_token_id", "eos_token_id",
            "bos_token_id", "pad_token_id", "length_penalty", "suppress_tokens", "begin_suppress_tokens")
    out = {k: cfg.get(k) for k in keep if k in cfg}
    out["call_overrides"] = gen_kwargs
    return out


class InputMismatch(RuntimeError):
    pass


class TokenCountMismatch(RuntimeError):
    pass


class NonfiniteOutput(RuntimeError):
    pass
