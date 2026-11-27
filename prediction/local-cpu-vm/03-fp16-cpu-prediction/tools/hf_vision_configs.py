#!/usr/bin/env python3
"""Step 2 (2026-11-23): config.json of every transformers image-classification repo in hf_image_cls_list.jsonl
(public files only, no token, no weights). id2label/label2id dropped (classifier labels do not change conv shapes).
Output: results/conv_catalog/hf_configs.jsonl (resumable)."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

E = Path(__file__).resolve().parents[1]
D = E / "results/conv_catalog"
OUT = D / "hf_configs.jsonl"
H = {"User-Agent": "cgf-research-client"}
rows = [json.loads(x) for x in (D / "hf_image_cls_list.jsonl").read_text().splitlines()]
todo = [r for r in rows if r["library"] == "transformers"]
done = {json.loads(x)["id"] for x in OUT.read_text().splitlines()} if OUT.exists() else set()
todo = [r for r in todo if r["id"] not in done]


def get(r):
    url = f"https://huggingface.co/{r['id']}/resolve/main/config.json"
    for k in range(6):
        try:
            q = requests.get(url, headers=H, timeout=30)
        except requests.RequestException:
            time.sleep(5 * (k + 1)); continue
        if q.status_code == 429 or q.status_code >= 500:
            time.sleep(15 * (k + 1)); continue
        if q.status_code != 200:
            return {"id": r["id"], "error": q.status_code}
        try:
            c = q.json()
        except ValueError:
            return {"id": r["id"], "error": "bad_json"}
        for key in ("id2label", "label2id"):
            c.pop(key, None)
        return {"id": r["id"], "downloads": r["downloads"], "config": c}
    return {"id": r["id"], "error": "retries"}


print("todo", len(todo), flush=True)
with ThreadPoolExecutor(6) as ex, OUT.open("a") as f:
    for i, res in enumerate(ex.map(get, todo), 1):
        f.write(json.dumps(res) + "\n")
        if i % 1000 == 0:
            f.flush(); print(i, flush=True)
print("HF_CONFIGS_DONE", flush=True)
