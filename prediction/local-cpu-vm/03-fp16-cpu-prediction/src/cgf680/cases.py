"""Common 680-condition table (FP16) + 26-case pilot subset.

Run:  python -m cgf680.cases --out manifest/cases.jsonl
The same table must be used by the GPU runner; case_id is the join key.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

BATCHES = [1, 8, 16, 32, 128]
DTYPE = "fp16"

MODELS = [
    # (model_id, short, kind, axis values)
    ("facebook/opt-125m", "opt-125m", "decoder_llm", [32, 128, 512, 1024]),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tinyllama-1.1b", "decoder_llm", [32, 128, 512, 1024]),
    ("Qwen/Qwen2.5-1.5B-Instruct", "qwen2.5-1.5b", "decoder_llm", [32, 128, 512, 1024]),
    ("openai-community/gpt2", "gpt2", "decoder_llm", [32, 128, 512]),
    ("openai-community/gpt2-medium", "gpt2-medium", "decoder_llm", [32, 128, 512]),
    ("openai-community/gpt2-xl", "gpt2-xl", "decoder_llm", [32, 128, 512]),
    ("microsoft/resnet-18", "resnet-18", "vision_cls", [128, 224, 256, 512]),
    ("microsoft/resnet-50", "resnet-50", "vision_cls", [128, 224, 256, 512]),
    ("google/mobilenet_v2_1.0_224", "mobilenet-v2", "vision_cls", [128, 224, 256, 512]),
    ("google/vit-large-patch16-224", "vit-large", "vision_cls", [224]),
    ("google-t5/t5-small", "t5-small", "text_encdec", [32, 128, 512, 1024]),
    ("google-t5/t5-base", "t5-base", "text_encdec", [32, 128, 512, 1024]),
    ("google-t5/t5-3b", "t5-3b", "text_encdec", [32, 128, 512, 1024]),
]

# Pilot (same 26 as the GPU pilot): text (8,128,128),(16,512,512); variable-res vision (8,224),(16,512);
# ViT (8,224),(16,224).
def is_pilot(c):
    if c["kind"] in ("decoder_llm", "text_encdec"):
        return (c["batch_size"], c["input_len"], c["new_tokens"]) in {(8, 128, 128), (16, 512, 512)}
    if c["short"] == "vit-large":
        return (c["batch_size"], c["resolution"]) in {(8, 224), (16, 224)}
    return (c["batch_size"], c["resolution"]) in {(8, 224), (16, 512)}


def build():
    cases = []
    for model, short, kind, axis in MODELS:
        if kind == "vision_cls":
            for b, r in itertools.product(BATCHES, axis):
                cases.append({"case_id": f"{DTYPE}__{short}__b{b}__r{r}", "model": model, "short": short, "kind": kind,
                              "dtype": DTYPE, "batch_size": b, "resolution": r, "input_len": None, "new_tokens": None})
        else:
            for b, l, n in itertools.product(BATCHES, axis, axis):
                tag = "enc" if kind == "text_encdec" else "in"
                cases.append({"case_id": f"{DTYPE}__{short}__b{b}__{tag}{l}__new{n}", "model": model, "short": short,
                              "kind": kind, "dtype": DTYPE, "batch_size": b, "resolution": None, "input_len": l,
                              "new_tokens": n})
    for i, c in enumerate(cases):
        c["index"] = i
        c["pilot"] = is_pilot(c)
    return cases


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="manifest/cases.jsonl")
    a = p.parse_args()
    cases = build()
    ids = [c["case_id"] for c in cases]
    assert len(ids) == len(set(ids)) == 680, len(ids)
    assert sum(c["pilot"] for c in cases) == 26, sum(c["pilot"] for c in cases)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(c, sort_keys=True) + "\n" for c in cases)
    out.write_text(text)
    cols = ["index", "case_id", "model", "short", "kind", "dtype", "batch_size", "input_len", "new_tokens", "resolution", "pilot"]
    with out.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for c in cases:
            w.writerow({k: c[k] for k in cols})
    per_model = {}
    for c in cases:
        per_model[c["model"]] = per_model.get(c["model"], 0) + 1
    print(json.dumps({"cases": len(cases), "pilot": sum(c["pilot"] for c in cases), "per_model": per_model,
                      "sha256": hashlib.sha256(text.encode()).hexdigest()}, indent=1))


if __name__ == "__main__":
    main()
