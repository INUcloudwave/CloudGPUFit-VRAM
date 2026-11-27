#!/usr/bin/env python3
"""Which CUDA attention kernels run under the transformers DEFAULT attention (sdpa where supported), FP16.

Separate diagnostic process (not a VRAM measurement). For each listed condition (same model/input recipe as the
worker): profile generate(max_new_tokens=1) = prefill only, and generate(max_new_tokens=2) = prefill + 1 decode step.
Records CUDA kernel names (all) and the attention-related subset, so the CPU side can see which SDPA backend
(flash / memory-efficient / math / cudnn) the GPU actually used for prefill and for decode.

  python -I diag_attention_kernels.py /config /out/attention_kernels.json id1 id2 ...
"""
import json
import re
import sys
from pathlib import Path

import torch
import transformers
from torch.profiler import ProfilerActivity, profile

cfgdir, out_path, ids = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
models = json.loads((cfgdir / "models.json").read_text())
conds = {c["condition_id"]: c for c in json.loads((cfgdir / "conditions.json").read_text())["conditions"]}
PAT = re.compile(r"flash|fmha|efficient|attention|attn|cutlass.*(mem|attention)|softmax|sdpa|cudnn", re.I)


def kernels(fn):
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    names = {}
    for e in p.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            names[e.name] = names.get(e.name, 0) + 1
    return names


rows = []
for cid in ids:
    c = conds[cid]
    pin = models["models"][c["model"]]
    cls = getattr(transformers, pin["loader"])
    m = cls.from_pretrained(c["model"], revision=pin["revision"], torch_dtype=torch.float16, low_cpu_mem_usage=True,
                            device_map={"": "cuda:0"}, use_safetensors=pin["use_safetensors"],
                            local_files_only=True).eval()
    cfg = m.config
    g = torch.Generator().manual_seed(models["common"]["seed"])
    if c["workload_type"] == "vision_cls":
        r = c["resolution"]
        px = torch.randn(c["batch_size"], 3, r, r, generator=g, dtype=torch.float32).to(torch.float16).cuda()

        def f():
            with torch.inference_mode():
                m(pixel_values=px)
        f()
        k1 = kernels(f)
        rows.append({"condition_id": cid, "model": c["model"], "attn_implementation": cfg._attn_implementation,
                     "forward_attention_kernels": {k: v for k, v in k1.items() if PAT.search(k)},
                     "forward_all_kernels": k1})
        print(json.dumps({k: rows[-1][k] for k in ("condition_id", "attn_implementation", "forward_attention_kernels")}),
              flush=True)
        del m
        torch.cuda.empty_cache()
        continue
    ids_ = torch.randint(3, cfg.vocab_size, (c["batch_size"], c["input_len"]), generator=g, dtype=torch.int64)
    for name in ("pad_token_id", "eos_token_id", "bos_token_id", "decoder_start_token_id"):
        v = getattr(cfg, name, None)
        for t in (v if isinstance(v, list) else [v] if isinstance(v, int) else []):
            ids_[ids_ == t] = 3
    inp = {"input_ids": ids_.cuda(), "attention_mask": torch.ones_like(ids_).cuda()}
    eos = cfg.eos_token_id[0] if isinstance(cfg.eos_token_id, list) else cfg.eos_token_id
    gen = transformers.GenerationConfig.from_model_config(cfg)
    gen.update(do_sample=False, num_beams=1, use_cache=True,
               pad_token_id=cfg.pad_token_id if cfg.pad_token_id is not None else eos)
    m.generation_config = gen

    def g1():
        with torch.inference_mode():
            m.generate(**inp, min_new_tokens=1, max_new_tokens=1)

    def g2():
        with torch.inference_mode():
            m.generate(**inp, min_new_tokens=2, max_new_tokens=2)
    g2()  # warm (lazy module load, cuBLAS init)
    k1, k2 = kernels(g1), kernels(g2)
    decode = {k: v - k1.get(k, 0) for k, v in k2.items() if v - k1.get(k, 0) > 0}
    row = {"condition_id": cid, "model": c["model"], "attn_implementation": cfg._attn_implementation,
           "prefill_attention_kernels": {k: v for k, v in k1.items() if PAT.search(k)},
           "decode_step_attention_kernels": {k: v for k, v in decode.items() if PAT.search(k)},
           "prefill_all_kernels": k1, "decode_step_all_kernels": decode}
    rows.append(row)
    print(json.dumps({k: row[k] for k in ("condition_id", "attn_implementation", "prefill_attention_kernels",
                                         "decode_step_attention_kernels")}), flush=True)
    del m
    torch.cuda.empty_cache()
json.dump({"torch": torch.__version__, "transformers": transformers.__version__,
           "sdpa_enabled": {"flash": torch.backends.cuda.flash_sdp_enabled(),
                            "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                            "math": torch.backends.cuda.math_sdp_enabled(),
                            "cudnn": torch.backends.cuda.cudnn_sdp_enabled()},
           "note": "decode = kernels(generate 2 tokens) - kernels(generate 1 token); separate process, not a measurement",
           "rows": rows}, open(out_path, "w"), indent=1)
