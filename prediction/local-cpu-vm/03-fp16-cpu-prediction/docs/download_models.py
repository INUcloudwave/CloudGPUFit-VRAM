"""Download pinned config + weights into HF_HOME (run with network). Usage: download_models.py pins.json [model ...]"""
import json, sys
from huggingface_hub import snapshot_download
pins = json.load(open(sys.argv[1])); only = sys.argv[2:]
for m, p in pins.items():
    if only and m not in only: continue
    d = snapshot_download(m, revision=p["revision"], allow_patterns=["config.json", "generation_config.json", "preprocessor_config.json", p["weights_file"]])
    print("OK", m, p["revision"], d, flush=True)
