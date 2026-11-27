"""Download the 13 pinned model snapshots into /exp/hf_cache/hub (run inside the image, WITH network, once).
File list and revisions = aws/model_files.json (taken from the INUOS cache, 28 GB). LFS blobs are verified by name
(the hub stores LFS blobs under their sha256)."""
import json, os, sys
from pathlib import Path
from huggingface_hub import hf_hub_download

spec = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else "/exp/aws/model_files.json").read_text())  # optional list (2026-11-23)
cache = "/exp/hf_cache/hub"
bad = []
for repo, m in spec.items():
    for f in m["files"]:
        p = hf_hub_download(repo, f["path"], revision=m["revision"], cache_dir=cache)
        sz = os.path.getsize(os.path.realpath(p))
        ok = sz == f["size"] and (not f.get("blob") or Path(os.path.realpath(p)).name == f["blob"])
        print(("OK  " if ok else "BAD ") + f"{repo}@{m['revision'][:10]} {f['path']} {sz}", flush=True)
        if not ok:
            bad.append((repo, f["path"]))
print("DOWNLOAD_DONE bad=", bad, flush=True)
sys.exit(1 if bad else 0)
