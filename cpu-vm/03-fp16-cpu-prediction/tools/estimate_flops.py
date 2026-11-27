#!/usr/bin/env python3
"""Rough per-case CPU time estimate (4 invocations) from FLOPs and measured FP16 kernel rates.
Rates (GFLOPS, 20 threads, this VM): linear/addmm ~70, bmm ~1.8 (no thread scaling), conv ~10.
Only for planning; replace with pilot-calibrated constants (--calib) once available."""
import json, sys
from pathlib import Path
EXP = Path("/home/inuos/cgf-fp16-680-20261111")
# model: (n_layers, d_model, heads_for_attention, head_dim, nonemb_params, vocab, kind-specific)
M = {"opt-125m": (12, 768, 12, 64, 85e6, 50272), "tinyllama-1.1b": (22, 2048, 32, 64, 0.97e9, 32000),
     "qwen2.5-1.5b": (28, 1536, 12, 128, 1.31e9, 151936), "gpt2": (12, 768, 12, 64, 85e6, 50257),
     "gpt2-medium": (24, 1024, 16, 64, 302e6, 50257), "gpt2-xl": (48, 1600, 25, 64, 1.47e9, 50257),
     "t5-small": (6, 512, 8, 64, 35e6, 32128), "t5-base": (12, 768, 12, 64, 198e6, 32128),
     "t5-3b": (24, 1024, 32, 128, 2.85e9, 32128)}
GF_VIS = {"resnet-18": 1.82, "resnet-50": 4.1, "mobilenet-v2": 0.3, "vit-large": 61.6}  # GFLOP/img @224 (MAC*2)
R_LIN, R_BMM, R_CONV = 70e9, 1.8e9, 10e9
def est(c, rlin=R_LIN, rbmm=R_BMM, rconv=R_CONV):
    b = c["batch_size"]
    if c["kind"] == "vision_cls":
        f = GF_VIS[c["short"]] * 1e9 * b * (c["resolution"] / 224) ** 2
        return 4 * f / (rlin if c["short"] == "vit-large" else rconv)
    L, H, Hh, dh, P, V = M[c["short"]]
    l, n = c["input_len"], c["new_tokens"]
    if c["kind"] == "decoder_llm":
        lin = 2 * P * b * (l + n) + 2 * V * L * 0 + 2 * V * H * b * n
        att = 4 * b * Hh * dh * L * (l * l + sum(range(l, l + n)))
    else:  # t5: P split enc/dec ~ 1/2 each; decoder has self+cross attention
        lin = 2 * (P / 2) * b * l + 2 * (P / 2) * b * n + 2 * V * H * b * n
        att = 4 * b * Hh * dh * L * (l * l + sum(range(1, n + 1)) + n * l)
    return 4 * (lin / rlin + att / rbmm)
if __name__ == "__main__":
    cases = [json.loads(x) for x in (EXP / "manifest/cases.jsonl").read_text().splitlines()]
    tot = 0; per = {}
    rows = []
    for c in cases:
        s = est(c); tot += s; per[c["short"]] = per.get(c["short"], 0) + s; rows.append((s, c["case_id"]))
    print("total days (sequential):", round(tot / 86400, 1))
    for k, v in sorted(per.items(), key=lambda x: -x[1]): print(f"  {k:16s} {v/86400:8.1f} d")
    rows.sort()
    import bisect
    for lim_h in (1, 4, 12, 24):
        ok = [s for s, _ in rows if s <= lim_h * 3600]
        print(f"cases <= {lim_h} h: {len(ok)}  (sum {sum(ok)/86400:.1f} d)")
    pil = [c for c in cases if c["pilot"]]
    print("pilot est hours:", round(sum(est(c) for c in pil) / 3600, 1))
    for c in pil: print(f"  {c['case_id']:42s} {est(c)/3600:7.2f} h")
