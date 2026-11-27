"""Build a 395-case manifest (jsonl) from a GPU condition table and pilot.json. case_id == GPU condition_id (join key).

Default (no arguments): conditions_main395.json (sha 2634bda7…) -> cases395.jsonl, exactly as before.
L256 table (2026-11-15):  --conditions conditions_main395_l256.json --out cases395_l256.jsonl
                          --expect-sha d4a9b09caff3064c00c04358bf2e462996bced23de93da98652bd1aba04b7289
"""
import argparse
import hashlib
import json
from pathlib import Path

M = Path(__file__).resolve().parents[2] / "manifest"
ap = argparse.ArgumentParser()
ap.add_argument("--conditions", default="conditions_main395.json", help="file name under manifest/")
ap.add_argument("--out", default="cases395.jsonl", help="file name under manifest/")
ap.add_argument("--expect-sha", default=None, help="abort unless the condition table has this sha256")
a = ap.parse_args()
raw = (M / a.conditions).read_bytes()
sha = hashlib.sha256(raw).hexdigest()
if a.expect_sha and sha != a.expect_sha:
    raise SystemExit(f"{a.conditions}: sha256 {sha} != expected {a.expect_sha}")
conds = json.loads(raw)["conditions"]
pilot = set(json.loads((M / "pilot.json").read_text())["condition_ids"])
models = json.loads((M / "models.json").read_text())["models"]
out = []
for i, c in enumerate(conds):
    out.append({"index": i, "case_id": c["condition_id"], "model": c["model"], "short": models[c["model"]]["short"],
                "kind": c["workload_type"], "dtype": "fp16", "batch_size": c["batch_size"], "input_len": c.get("input_len"),
                "new_tokens": c.get("max_new_tokens"), "resolution": c.get("resolution"), "pilot": c["condition_id"] in pilot})
assert len(out) == 395 and sum(x["pilot"] for x in out) == 26, (len(out), sum(x["pilot"] for x in out))
assert len({x["case_id"] for x in out}) == 395
(M / a.out).write_text("".join(json.dumps(x) + "\n" for x in out))
print(json.dumps({"cases": len(out), "pilot": 26, "conditions": a.conditions, "conditions_sha256": sha,
                  "models_sha256": hashlib.sha256((M / "models.json").read_bytes()).hexdigest(), "out": a.out}))
