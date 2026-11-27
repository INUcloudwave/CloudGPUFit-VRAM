#!/usr/bin/env python3
"""Diagnostic (separate from measurements): where/why Qwen2.5-1.5B-Instruct FP16 forward becomes non-finite.

One prefill forward per variant with the pilot input recipe (B=8, L=128, seed 20261006).
Variants: dtype (fp16/bf16/fp32) x attention (eager/sdpa) x input id range (config.vocab_size / tokenizer size).
For each: first module (in execution order) whose output has non-finite values, and max |activation| per layer.
"""
import json
import sys
import torch
import transformers
from transformers import AutoModelForCausalLM

MODEL, REV = "Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
B, L, SEED = 8, 128, 20261006
TOK_VOCAB = 151665  # len(tokenizer) of this revision (tokenizer.json: 151643 base + 22 added)


def make_ids(cfg, hi):
    g = torch.Generator().manual_seed(SEED)
    ids = torch.randint(3, hi, (B, L), generator=g, dtype=torch.int64)
    for name in ("pad_token_id", "eos_token_id", "bos_token_id"):
        v = getattr(cfg, name, None)
        for t in (v if isinstance(v, list) else [v] if isinstance(v, int) else []):
            ids[ids == t] = 3
    return ids


def run(dtype, attn, hi_name):
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, torch_dtype=dtype, device_map={"": "cuda:0"},
                                                 local_files_only=True, attn_implementation=attn).eval()
    cfg = model.config
    hi = cfg.vocab_size if hi_name == "config_vocab" else TOK_VOCAB
    ids = make_ids(cfg, hi)
    first_bad, layer_max = [], {}

    def hook(name):
        def f(m, a, o):
            t = o[0] if isinstance(o, tuple) else o
            if not torch.is_tensor(t):
                return
            if not first_bad and not torch.isfinite(t).all():
                first_bad.append(name)
            if name.count(".") == 2 and name.startswith("model.layers."):   # decoder layer outputs
                layer_max[name] = float(t.float().abs().nan_to_num(posinf=float("inf")).max())
        return f

    hs = [m.register_forward_hook(hook(n)) for n, m in model.named_modules() if n]
    with torch.inference_mode():
        out = model(input_ids=ids.cuda(), attention_mask=torch.ones_like(ids).cuda(), use_cache=True)
    fin = bool(torch.isfinite(out.logits).all())
    for h in hs:
        h.remove()
    big = sorted(layer_max.items(), key=lambda kv: -kv[1])[:3]
    res = {"dtype": str(dtype), "attn": attn, "input_ids_max_exclusive": hi, "ids_max": int(ids.max()),
           "logits_all_finite": fin, "first_nonfinite_module": first_bad[0] if first_bad else None,
           "top3_layer_output_absmax": big, "fp16_max": 65504.0}
    del model, out
    torch.cuda.empty_cache()
    return res


rows = []
for dtype in (torch.float16, torch.bfloat16, torch.float32):
    for attn in ("eager", "sdpa"):
        for hi in ("config_vocab", "tokenizer_vocab"):
            if dtype != torch.float16 and hi == "tokenizer_vocab":
                continue
            r = run(dtype, attn, hi)
            rows.append(r)
            print(json.dumps(r), flush=True)
json.dump({"torch": torch.__version__, "transformers": transformers.__version__, "rows": rows},
          open(sys.argv[1], "w"), indent=1)
