#!/usr/bin/env python3
"""Exact relative file list (under HF_HOME) of the 13 pinned models, for streaming the cache with tar -T.

  python3 -I model_cache_filelist.py <config_dir> <hf_home>  > config/model_cache_filelist.txt
Lists, per manifest file: snapshot symlink + its blob target; plus refs/* that point to the pinned revision.
"""
import json
import os
import sys
from pathlib import Path

cfg, hf = Path(sys.argv[1]), Path(sys.argv[2])
man = json.loads((cfg / "model_cache_manifest.json").read_text())
out = []
for model, ent in man["models"].items():
    d = Path("hub") / f"models--{model.replace('/', '--')}"
    for ref in sorted((hf / d / "refs").glob("*")) if (hf / d / "refs").exists() else []:
        if ref.read_text().strip() == ent["revision"]:
            out.append(str(ref.relative_to(hf)))
    for f in ent["files"]:
        snap = d / "snapshots" / ent["revision"] / f["path"]
        assert (hf / snap).is_symlink(), snap
        blob = d / "blobs" / f["blob"]
        assert (hf / snap).resolve() == (hf / blob).resolve(), snap
        out += [str(snap), str(blob)]
print("\n".join(out))
