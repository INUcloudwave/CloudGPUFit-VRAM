#!/usr/bin/env python3
"""Build/verify the pinned 13-model HF cache on a new host (AWS). Run inside the common image WITH network:

  docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e HF_HOME=/hf -v <cache>:/hf -v <exp>:/exp:ro \
      --entrypoint python cloudgpufit-bench:cu124-v1 -I /exp/tools/prepare_cache.py /exp/config [--verify-only] [--hash]

--hash: additionally hash every file's content and compare with its blob name (64 hex = LFS sha256 of content,
        40 hex = git blob sha1 of "blob <size>\0" + content). Also works on the host (python3 -I, HF_HOME=<cache>).

Downloads exactly the files listed in config/model_cache_manifest.json at the pinned revisions and verifies
size + blob name (sha256 for LFS files) against the RTX 3090 cache. Exit 1 on any mismatch.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

cfg = Path(sys.argv[1])
verify_only = "--verify-only" in sys.argv
do_hash = "--hash" in sys.argv


def content_ok(path, blob, size):
    h = hashlib.sha256() if len(blob) == 64 else hashlib.sha1(b"blob %d\0" % size)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest() == blob


n_hashed = 0
man = json.loads((cfg / "model_cache_manifest.json").read_text())
hub = Path(os.environ.get("HF_HOME", "/hf")) / "hub"
bad = []
for model, ent in man["models"].items():
    rev = ent["revision"]
    if not verify_only:
        from huggingface_hub import hf_hub_download
        for f in ent["files"]:
            hf_hub_download(model, f["path"], revision=rev)
    snap = hub / f"models--{model.replace('/', '--')}" / "snapshots" / rev
    for f in ent["files"]:
        p = snap / f["path"]
        if not p.exists():
            bad.append((model, f["path"], "missing"))
            continue
        t = p.resolve()
        if t.stat().st_size != f["size"] or t.name != f["blob"]:
            bad.append((model, f["path"], f"size/blob {t.stat().st_size}/{t.name}"))
        elif do_hash:
            n_hashed += 1
            if not content_ok(t, f["blob"], f["size"]):
                bad.append((model, f["path"], "content hash != blob name"))
    print("checked", model, rev, flush=True)
print(json.dumps({"mismatches": bad, "ok": not bad, "files": sum(len(e["files"]) for e in man["models"].values()),
                  "content_hashed": n_hashed, "total_bytes": man.get("total_bytes")}))
sys.exit(1 if bad else 0)
