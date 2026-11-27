#!/usr/bin/env python3
"""Derive the main-experiment manifest (395 conditions by the per-group rules; the request text said 355, but its own
group totals 135+60+135+60+5 = 395 -- recorded, not silently changed) from the 680-condition table (subset; condition objects copied verbatim).

  python3 -I generate_main.py <config_dir>

Rules (2026-11-11): batches 1/8/16/32/128 for every model. Text length/new-token candidate 32 removed:
  OPT-125M/TinyLlama/Qwen2.5 and T5-small/base/3b: input x new in {128,512,1024}^2 ; GPT-2/medium/xl: {128,512}^2
  ResNet-18/50, MobileNet V2: resolution 128/224/256/512 ; ViT Large: 224 fixed.
Model ids, revisions, input/generation recipe unchanged (models.json is shared, not rewritten).
Outputs: conditions_main395.json (same schema as conditions.json), conditions_main395.csv
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

cfg = Path(sys.argv[1])
src = json.loads((cfg / "conditions.json").read_text())
models = json.loads((cfg / "models.json").read_text())["models"]
BATCHES = {1, 8, 16, 32, 128}
L3, L2, RES = {128, 512, 1024}, {128, 512}, {128, 224, 256, 512}
SHORT_RULE = {"opt-125m": L3, "tinyllama-1.1b-chat": L3, "qwen2.5-1.5b-instruct": L3,
              "t5-small": L3, "t5-base": L3, "t5-3b": L3, "gpt2": L2, "gpt2-medium": L2, "gpt2-xl": L2,
              "resnet-18": RES, "resnet-50": RES, "mobilenet-v2-1.0-224": RES, "vit-large-patch16-224": {224}}
EXPECT = {"opt-125m": 45, "tinyllama-1.1b-chat": 45, "qwen2.5-1.5b-instruct": 45, "t5-small": 45, "t5-base": 45,
          "t5-3b": 45, "gpt2": 20, "gpt2-medium": 20, "gpt2-xl": 20, "resnet-18": 20, "resnet-50": 20,
          "mobilenet-v2-1.0-224": 20, "vit-large-patch16-224": 5}


def keep(c):
    allowed = SHORT_RULE[models[c["model"]]["short"]]
    if c["batch_size"] not in BATCHES:
        return False
    if c["workload_type"] == "vision_cls":
        return c["resolution"] in allowed
    return c["input_len"] in allowed and c["max_new_tokens"] in allowed


kept = [c for c in src["conditions"] if keep(c)]
ids = [c["condition_id"] for c in kept]
assert len(ids) == 395 == len(set(ids)), len(ids)
per = {}
for c in kept:
    s = models[c["model"]]["short"]
    per[s] = per.get(s, 0) + 1
assert per == EXPECT, per
GROUPS = {"opt/tinyllama/qwen": ("opt-125m", "tinyllama-1.1b-chat", "qwen2.5-1.5b-instruct"),
          "gpt2 x3": ("gpt2", "gpt2-medium", "gpt2-xl"), "t5 x3": ("t5-small", "t5-base", "t5-3b"),
          "resnet/mobilenet": ("resnet-18", "resnet-50", "mobilenet-v2-1.0-224"), "vit": ("vit-large-patch16-224",)}
group_tot = {g: sum(per[m] for m in ms) for g, ms in GROUPS.items()}
assert group_tot == {"opt/tinyllama/qwen": 135, "gpt2 x3": 60, "t5 x3": 135, "resnet/mobilenet": 60, "vit": 5}, group_tot
# independent re-enumeration of the expected id set (same id format as generate_conditions.py)
exp = set()
for m, ent in models.items():
    s = ent["short"]
    for b in BATCHES:
        if ent["workload_type"] == "vision_cls":
            exp |= {f"{s}__b{b}__r{r}" for r in SHORT_RULE[s]}
        else:
            exp |= {f"{s}__b{b}__in{L}__new{n}" for L in SHORT_RULE[s] for n in SHORT_RULE[s]}
assert exp == set(ids), (exp ^ set(ids))
pilot = json.loads((cfg / "pilot.json").read_text())["condition_ids"]
missing = [p for p in pilot if p not in set(ids)]
assert not missing, missing
src_by_id = {c["condition_id"]: c for c in src["conditions"]}
assert all(src_by_id[c["condition_id"]] == c for c in kept)
(cfg / "conditions_main395.json").write_text(json.dumps({
    "schema": "cgf-fp16-vram-conditions/1", "count": len(kept), "derived_from": "conditions.json",
    "derived_from_sha256": hashlib.sha256((cfg / "conditions.json").read_bytes()).hexdigest(),
    "rule": "batches 1/8/16/32/128; text candidate 32 removed; vision unchanged; see generate_main355.py",
    "conditions": kept}, indent=1))
with (cfg / "conditions_main395.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, ["condition_id", "model", "workload_type", "batch_size", "input_len", "max_new_tokens",
                           "resolution"])
    w.writeheader()
    w.writerows(kept)
print(json.dumps({"group_totals": group_tot, "count": len(kept), "unique": len(set(ids)), "per_model": per,
                  "pilot_26_included": len(pilot) - len(missing), "dropped_from_680": len(src["conditions"]) - len(kept)}))
for n in ("conditions_main395.json", "conditions_main395.csv", "conditions.json", "models.json", "pilot.json"):
    print(hashlib.sha256((cfg / n).read_bytes()).hexdigest(), n)
