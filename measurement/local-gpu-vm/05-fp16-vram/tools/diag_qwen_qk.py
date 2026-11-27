#!/usr/bin/env python3
"""Max |Q K^T| (before scaling, as eager computes it) per layer, in fp32, for Qwen2.5-1.5B-Instruct.
Also for B=1 L=32 to see whether overflow is input-size independent."""
import json, sys, torch
import transformers.models.qwen2.modeling_qwen2 as q2
from transformers import AutoModelForCausalLM
MODEL, REV = "Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
rec = []
orig = q2.eager_attention_forward
def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    k = q2.repeat_kv(key, module.num_key_value_groups)
    raw = torch.matmul(query.float(), k.float().transpose(2, 3))
    rec.append({"layer": module.layer_idx, "max_abs_qk_unscaled": float(raw.abs().max()),
                "max_abs_qk_scaled": float((raw * scaling).abs().max()), "scaling": scaling})
    return orig(module, query, key, value, attention_mask, scaling, dropout, **kw)
q2.eager_attention_forward = patched
m = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, torch_dtype=torch.float32, device_map={"": "cuda:0"},
                                         local_files_only=True, attn_implementation="eager").eval()
out = {}
for B, L in ((8, 128), (1, 32)):
    rec.clear()
    g = torch.Generator().manual_seed(20261006)
    ids = torch.randint(3, m.config.vocab_size, (B, L), generator=g)
    with torch.inference_mode():
        m(input_ids=ids.cuda(), attention_mask=torch.ones_like(ids).cuda())
    over = [r["layer"] for r in rec if r["max_abs_qk_unscaled"] > 65504]
    out[f"B{B}_L{L}"] = {"layers_over_fp16_max_unscaled": over, "per_layer": rec[:]}
    print(f"B{B} L{L} overflow layers(unscaled>65504):", over)
    print("  top:", sorted(((r["layer"], round(r["max_abs_qk_unscaled"]), round(r["max_abs_qk_scaled"])) for r in rec), key=lambda x: -x[1])[:5])
json.dump(out, open(sys.argv[1], "w"), indent=1)
