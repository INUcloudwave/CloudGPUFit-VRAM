#!/usr/bin/env python3
"""Generate the common 680-condition table + 26-condition RTX 3090 pilot list (deterministic, no randomness).

  python3 -I generate_conditions.py <out_dir>

Outputs: models.json (pinned revision + loader per model), conditions.json, conditions.csv, pilot.json
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

BATCHES = [1, 8, 16, 32, 128]
TEXT4 = [32, 128, 512, 1024]
TEXT3 = [32, 128, 512]
RES4 = [128, 224, 256, 512]

# model_id: (short, workload, revision, use_safetensors, lengths/resolutions)
MODELS = {
    "facebook/opt-125m": ("opt-125m", "decoder_llm", "27dcfa74d334bc871f3234de431e71c6eeba5dd6", False, TEXT4),
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": ("tinyllama-1.1b-chat", "decoder_llm", "fe8a4ea1ffedaf415f4da2f062534de366a451e6", True, TEXT4),
    "Qwen/Qwen2.5-1.5B-Instruct": ("qwen2.5-1.5b-instruct", "decoder_llm", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306", True, TEXT4),
    "openai-community/gpt2": ("gpt2", "decoder_llm", "607a30d783dfa663caf39e06633721c8d4cfcd7e", True, TEXT3),
    "openai-community/gpt2-medium": ("gpt2-medium", "decoder_llm", "6dcaa7a952f72f9298047fd5137cd6e4f05f41da", True, TEXT3),
    "openai-community/gpt2-xl": ("gpt2-xl", "decoder_llm", "15ea56dee5df4983c59b2538573817e1667135e2", True, TEXT3),
    "microsoft/resnet-18": ("resnet-18", "vision_cls", "65a5785d9156231087c481e0c7dd33a5ff6f7e3e", True, RES4),
    "microsoft/resnet-50": ("resnet-50", "vision_cls", "34c2154c194f829b11125337b98c8f5f9965ff19", True, RES4),
    "google/mobilenet_v2_1.0_224": ("mobilenet-v2-1.0-224", "vision_cls", "75e607b00aeae1297cc89d026a118bce012f5c5a", True, RES4),
    "google/vit-large-patch16-224": ("vit-large-patch16-224", "vision_cls", "9e2727f4250d3973839eecfa5c4b42e41b709a50", False, [224]),
    "google-t5/t5-small": ("t5-small", "text_encdec", "df1b051c49625cf57a3d0d8d3863ed4d13564fe4", True, TEXT4),
    "google-t5/t5-base": ("t5-base", "text_encdec", "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1", True, TEXT4),
    "google-t5/t5-3b": ("t5-3b", "text_encdec", "bed96aab9ee46012a5046386105ee5fd0ac572f0", True, TEXT4),
}
LOADER = {"decoder_llm": "AutoModelForCausalLM", "text_encdec": "AutoModelForSeq2SeqLM",
          "vision_cls": "AutoModelForImageClassification"}
PILOT = {"text": [(8, 128, 128), (16, 512, 512)], "vision": [(8, 224), (16, 512)], "vit": [(8, 224), (16, 224)]}

COMMON = {
    "dtype": "fp16",
    "seed": 20261006,
    "attn_implementation": "eager",
    "load_kwargs": {"torch_dtype": "fp16", "low_cpu_mem_usage": True, "device_map": {"": "cuda:0"},
                    "local_files_only": True, "attn_implementation": "eager"},
    "after_load": ".to(device='cuda', dtype=float16).eval(); config.output_attentions=False; config.output_hidden_states=False",
    "text_input_rule": ("g=torch.Generator().manual_seed(seed); input_ids=torch.randint(3, config.vocab_size, (B, L), "
                        "generator=g, dtype=int64); ids equal to config pad/eos/bos/decoder_start (int or list) -> 3; "
                        "attention_mask=torch.ones(B, L, dtype=int64). No tokenizer, no BOS insertion, no truncation."),
    "image_input_rule": ("g=torch.Generator().manual_seed(seed); pixel_values=torch.randn(B, 3, R, R, generator=g, "
                         "dtype=float32).to(float16). No image processor (no resize/crop)."),
    "generation_rule": ("GenerationConfig.from_model_config(model.config) then do_sample=False, num_beams=1, use_cache=True, "
                        "min_new_tokens=max_new_tokens=N, pad_token_id=config.pad_token_id or eos, no repetition penalty/"
                        "sampling knobs from hub generation_config.json. decoder: output.shape[1]-L == N; "
                        "T5: decoder starts from decoder_start_token_id only, output.shape[1]-1 == N (start token excluded)"),
    "inputs_device_rule": "inputs kept on host CPU, moved with .to('cuda') inside every invocation",
}


def conditions():
    out = []
    for model, (short, kind, rev, safe, vals) in MODELS.items():
        for b in BATCHES:
            if kind == "vision_cls":
                for r in vals:
                    out.append({"condition_id": f"{short}__b{b}__r{r}", "model": model, "workload_type": kind,
                                "batch_size": b, "resolution": r})
            else:
                for L in vals:
                    for n in vals:
                        out.append({"condition_id": f"{short}__b{b}__in{L}__new{n}", "model": model,
                                    "workload_type": kind, "batch_size": b, "input_len": L, "max_new_tokens": n})
    return out


def pilot_ids():
    ids = []
    for model, (short, kind, *_rest) in MODELS.items():
        if kind == "vision_cls":
            grid = PILOT["vit"] if short.startswith("vit") else PILOT["vision"]
            ids += [f"{short}__b{b}__r{r}" for b, r in grid]
        else:
            ids += [f"{short}__b{b}__in{L}__new{n}" for b, L, n in PILOT["text"]]
    return ids


def main():
    out = Path(sys.argv[1])
    conds = conditions()
    assert len(conds) == 680 and len({c["condition_id"] for c in conds}) == 680
    pil = pilot_ids()
    ids = {c["condition_id"] for c in conds}
    assert len(pil) == 26 and all(p in ids for p in pil)
    models = {m: {"short": s, "workload_type": k, "revision": r, "use_safetensors": safe, "loader": LOADER[k]}
              for m, (s, k, r, safe, _v) in MODELS.items()}
    (out / "models.json").write_text(json.dumps({"common": COMMON, "models": models}, indent=1))
    (out / "conditions.json").write_text(json.dumps({"schema": "cgf-fp16-vram-conditions/1", "count": len(conds),
                                                     "conditions": conds}, indent=1))
    (out / "pilot.json").write_text(json.dumps({"gpu": "rtx3090", "condition_ids": pil}, indent=1))
    with (out / "conditions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, ["condition_id", "model", "workload_type", "batch_size", "input_len", "max_new_tokens",
                               "resolution"])
        w.writeheader()
        w.writerows(conds)
    for n in ("models.json", "conditions.json", "pilot.json", "conditions.csv"):
        print(hashlib.sha256((out / n).read_bytes()).hexdigest(), n)


if __name__ == "__main__":
    main()
