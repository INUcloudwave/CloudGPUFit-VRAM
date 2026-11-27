#!/usr/bin/env python3
"""Step 1 of the conv-shape catalogue (2026-11-23): list public HF image-classification models (metadata only, no
token, no weights). Output: results/conv_catalog/hf_image_cls_list.jsonl (id, library, model_type tag, downloads)."""
import json
import re
import sys
import time
from pathlib import Path

import requests

E = Path(__file__).resolve().parents[1]
OUT = E / "results/conv_catalog/hf_image_cls_list.jsonl"
H = {"User-Agent": "cgf-research-client"}
url = "https://huggingface.co/api/models?pipeline_tag=image-classification&sort=downloads&direction=-1&limit=1000&expand[]=downloads&expand[]=library_name&expand[]=tags&expand[]=config"
n = 0
with OUT.open("w") as f:
    while url:
        for k in range(5):
            r = requests.get(url, headers=H, timeout=60)
            if r.status_code == 200:
                break
            time.sleep(10 * (k + 1))
        r.raise_for_status()
        for m in r.json():
            cfg = m.get("config") or {}
            f.write(json.dumps({"id": m["id"], "library": m.get("library_name"), "downloads": m.get("downloads", 0),
                                "model_type": cfg.get("model_type"), "architectures": cfg.get("architectures"),
                                "tags": [t for t in m.get("tags", []) if not t.startswith(("license:", "region:", "dataset:", "arxiv:", "base_model:"))]}) + "\n")
            n += 1
        lk = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", ""))
        url = lk.group(1) if lk else None
        print(n, flush=True)
        time.sleep(0.5)
print("HF_LIST_DONE", n)
