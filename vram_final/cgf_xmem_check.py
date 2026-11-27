"""xMem source check (from cpu-vm/02-reserved-diag/src/cgf_diag/replay.py verify_xmem): the allocator files must be
the pinned commit 1ee4bd08, unmodified, and the imported module must be the one under the given root."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def verify_xmem(xmem_root, lock_path):
    lock = json.loads(Path(lock_path).read_text())
    out = {"commit": lock["commit"], "files": {}}
    ok = True
    for rel, expected in lock["sha256"].items():
        f = Path(xmem_root) / rel
        got = hashlib.sha256(f.read_bytes()).hexdigest() if f.is_file() else None
        out["files"][rel] = got == expected
        ok &= got == expected
    import perf_estimator.allocator.allocator as mod
    out["imported_from"] = mod.__file__
    out["imported_matches_root"] = (Path(mod.__file__).resolve()
                                    == (Path(xmem_root) / "perf_estimator/allocator/allocator.py").resolve())
    out["ok"] = ok and out["imported_matches_root"]
    if not out["ok"]:
        raise RuntimeError(f"xMem source does not match the pinned commit {lock['commit'][:8]}: {out}")
    return out
