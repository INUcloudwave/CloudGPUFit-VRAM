#!/usr/bin/env python3
"""Main manifest v2 (2026-11-15): same 395 layout as conditions_main395.json, but text length/new-token candidate
1024 -> 256 for OPT-125M/TinyLlama/Qwen2.5 and T5-small/base/3b (reason: 1024 runs too long; user request).
  OPT/TinyLlama/Qwen, T5 x3: input x new in {128,256,512}^2 ; GPT-2 x3: {128,512}^2 (unchanged)
  ResNet-18/50, MobileNet V2: 128/224/256/512 ; ViT Large: 224 ; batches 1/8/16/32/128 (all unchanged)
256 is not in the 680 table, so those condition objects are built with the exact schema/key order of
generate_conditions.py; every condition that also exists in the 680 table must equal it byte-for-byte.
Model ids / revisions / recipe (models.json) unchanged. conditions_main395.json is kept as-is.

  python3 -I generate_main_l256.py <config_dir>
"""
import csv, hashlib, json, sys
from pathlib import Path

cfg = Path(sys.argv[1])
src = json.loads((cfg / "conditions.json").read_text())
models = json.loads((cfg / "models.json").read_text())["models"]
BATCHES = [1, 8, 16, 32, 128]
L3, L2, RES = [128, 256, 512], [128, 512], [128, 224, 256, 512]
RULE = {"opt-125m": L3, "tinyllama-1.1b-chat": L3, "qwen2.5-1.5b-instruct": L3, "t5-small": L3, "t5-base": L3,
        "t5-3b": L3, "gpt2": L2, "gpt2-medium": L2, "gpt2-xl": L2, "resnet-18": RES, "resnet-50": RES,
        "mobilenet-v2-1.0-224": RES, "vit-large-patch16-224": [224]}
out = []
for model, ent in models.items():  # same iteration order / object layout as generate_conditions.py
    s, kind = ent["short"], ent["workload_type"]
    for b in BATCHES:
        if kind == "vision_cls":
            out += [{"condition_id": f"{s}__b{b}__r{r}", "model": model, "workload_type": kind, "batch_size": b,
                     "resolution": r} for r in RULE[s]]
        else:
            out += [{"condition_id": f"{s}__b{b}__in{L}__new{n}", "model": model, "workload_type": kind,
                     "batch_size": b, "input_len": L, "max_new_tokens": n} for L in RULE[s] for n in RULE[s]]
ids = [c["condition_id"] for c in out]
assert len(ids) == 395 == len(set(ids)), len(ids)
per = {}
for c in out:
    per[models[c["model"]]["short"]] = per.get(models[c["model"]]["short"], 0) + 1
assert per == {"opt-125m": 45, "tinyllama-1.1b-chat": 45, "qwen2.5-1.5b-instruct": 45, "t5-small": 45, "t5-base": 45,
               "t5-3b": 45, "gpt2": 20, "gpt2-medium": 20, "gpt2-xl": 20, "resnet-18": 20, "resnet-50": 20,
               "mobilenet-v2-1.0-224": 20, "vit-large-patch16-224": 5}, per
assert not any(c.get("input_len") == 1024 or c.get("max_new_tokens") == 1024 for c in out)
src_by_id = {c["condition_id"]: c for c in src["conditions"]}
shared = [c for c in out if c["condition_id"] in src_by_id]
assert all(src_by_id[c["condition_id"]] == c and list(src_by_id[c["condition_id"]]) == list(c) for c in shared)
new = [c for c in out if c["condition_id"] not in src_by_id]
assert all(256 in (c["input_len"], c["max_new_tokens"]) for c in new)
old395 = {c["condition_id"] for c in json.loads((cfg / "conditions_main395.json").read_text())["conditions"]}
pilot = json.loads((cfg / "pilot.json").read_text())["condition_ids"]
assert set(pilot) <= set(ids)
(cfg / "conditions_main395_l256.json").write_text(json.dumps({
    "schema": "cgf-fp16-vram-conditions/1", "count": len(out), "derived_from": "conditions.json + rule (256 added)",
    "derived_from_sha256": hashlib.sha256((cfg / "conditions.json").read_bytes()).hexdigest(),
    "rule": "batches 1/8/16/32/128; OPT/TinyLlama/Qwen/T5 lengths {128,256,512}^2; GPT-2 {128,512}^2; vision unchanged; "
            "see generate_main_l256.py", "conditions": out}, indent=1))
with (cfg / "conditions_main395_l256.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, ["condition_id", "model", "workload_type", "batch_size", "input_len", "max_new_tokens",
                           "resolution"])
    w.writeheader(); w.writerows(out)
print(json.dumps({"count": len(out), "unique": len(set(ids)), "in_680_table": len(shared), "new_with_256": len(new),
                  "same_id_as_old395": len(old395 & set(ids)), "old395_dropped(1024)": len(old395 - set(ids)),
                  "pilot_26_included": len(pilot)}))
for n in ("conditions_main395_l256.json", "conditions_main395_l256.csv", "conditions_main395.json", "conditions.json",
          "models.json"):
    print(hashlib.sha256((cfg / n).read_bytes()).hexdigest(), n)
