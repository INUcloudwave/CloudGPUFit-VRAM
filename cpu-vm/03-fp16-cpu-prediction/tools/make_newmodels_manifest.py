#!/usr/bin/env python3
"""Generalisation test 2026-11-22: models not used so far (3 CNN + 3 LLM), same protocol as main395_l256.
Writes (stdlib only):
  manifest/models_newmodels_20261122.json      (schema of manifest/models.json; 'common' copied unchanged)
  manifest/model_pins_newmodels_20261122.json  (for docs/download_models.py)
  manifest/conditions_newmodels_20261122.json  (INUGPU condition schema cgf-fp16-vram-conditions/1)
  manifest/cases_newmodels_20261122.jsonl      (CPU case list, same fields as cases395_l256.jsonl)
Revisions: Hugging Face main at 2026-11-22 (pinned here)."""
import json
from pathlib import Path

E = Path(__file__).resolve().parents[1]
NEW = {
    "facebook/convnext-tiny-224": ("convnext-tiny-224", "vision_cls", "6166b7613034066690a621d8bf25ffdf181a34f0", "pytorch_model.bin"),
    "google/efficientnet-b0": ("efficientnet-b0", "vision_cls", "3d8d75d0812b65064f0151b456f869c6344bf20c", "pytorch_model.bin"),
    "facebook/regnet-y-040": ("regnet-y-040", "vision_cls", "29e70d26bce1b86bc90c32ac5fadedfc1b671285", "pytorch_model.bin"),
    "Qwen/Qwen2.5-0.5B-Instruct": ("qwen2.5-0.5b-instruct", "decoder_llm", "7ae557604adf67be50417f59c2c2f167def9a775", "model.safetensors"),
    "HuggingFaceTB/SmolLM2-360M-Instruct": ("smollm2-360m-instruct", "decoder_llm", "a10cc1512eabd3dde888204e902eca88bddb4951", "model.safetensors"),
    "EleutherAI/pythia-410m-deduped": ("pythia-410m-deduped", "decoder_llm", "c4fc8d586d62df497f1f9b69d66d3ca419992d3e", "model.safetensors"),
}
LOADER = {"decoder_llm": "AutoModelForCausalLM", "vision_cls": "AutoModelForImageClassification"}
BATCH = (1, 8, 32, 128)
LLM_LEN = (128, 512)          # input x new tokens {128,512}^2
RES = (128, 224, 256, 512)

base = json.loads((E / "manifest/models.json").read_text())
models = {"common": base["common"], "models": {}}
pins = {}
for m, (short, kind, rev, wf) in NEW.items():
    st = wf.endswith(".safetensors")
    models["models"][m] = {"short": short, "workload_type": kind, "revision": rev, "use_safetensors": st, "loader": LOADER[kind]}
    pins[m] = {"revision": rev, "weights_file": wf, "safetensors": st}
conds, cases = [], []
for m, (short, kind, rev, wf) in NEW.items():
    for b in BATCH:
        if kind == "decoder_llm":
            for L in LLM_LEN:
                for N in LLM_LEN:
                    cid = f"{short}__b{b}__in{L}__new{N}"
                    conds.append({"condition_id": cid, "model": m, "workload_type": kind, "batch_size": b, "input_len": L, "max_new_tokens": N})
                    cases.append({"case_id": cid, "model": m, "short": short, "kind": kind, "dtype": "fp16", "batch_size": b,
                                  "input_len": L, "new_tokens": N, "resolution": None, "pilot": False})
        else:
            for r in RES:
                cid = f"{short}__b{b}__r{r}"
                conds.append({"condition_id": cid, "model": m, "workload_type": kind, "batch_size": b, "resolution": r})
                cases.append({"case_id": cid, "model": m, "short": short, "kind": kind, "dtype": "fp16", "batch_size": b,
                              "input_len": None, "new_tokens": None, "resolution": r, "pilot": False})
for i, c in enumerate(cases):
    c["index"] = i
cj = {"schema": "cgf-fp16-vram-conditions/1", "count": len(conds), "derived_from": "INUOS tools/make_newmodels_manifest.py (2026-11-22)",
      "rule": "generalisation test: 3 CNN (convnext-tiny, efficientnet-b0, regnet-y-040) x batch 1/8/32/128 x resolution 128/224/256/512; "
              "3 LLM (qwen2.5-0.5b, smollm2-360m, pythia-410m) x batch 1/8/32/128 x input {128,512} x new {128,512}; same protocol as main395_l256",
      "conditions": conds}
(E / "manifest/models_newmodels_20261122.json").write_text(json.dumps(models, indent=1) + "\n")
(E / "manifest/model_pins_newmodels_20261122.json").write_text(json.dumps(pins, indent=1) + "\n")
(E / "manifest/conditions_newmodels_20261122.json").write_text(json.dumps(cj, indent=1) + "\n")
(E / "manifest/cases_newmodels_20261122.jsonl").write_text("".join(json.dumps({k: c[k] for k in ("index", "case_id", "model", "short", "kind", "dtype", "batch_size", "input_len", "new_tokens", "resolution", "pilot")}) + "\n" for c in cases))
print(len(conds), "conditions")
