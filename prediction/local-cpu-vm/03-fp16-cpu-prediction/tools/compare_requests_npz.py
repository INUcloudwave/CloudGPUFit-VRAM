#!/usr/bin/env python3
"""Exact equality of two worker-v2 request traces (invocation_fp16_requests.npz): action/size/alloc pairing of the
traced invocation, the output-alive marker position and the synthesized load-phase events.
  python3 tools/compare_requests_npz.py CASE_DIR_A CASE_DIR_B [...]   (first dir is the reference)"""
import json
import sys

import numpy as np


def canon_pairs(action, alloc_id):
    """alloc ids renumbered by first appearance -> pairing structure independent of raw id values."""
    m, out = {}, np.empty(len(alloc_id), dtype=np.int64)
    for i, x in enumerate(alloc_id.tolist()):
        out[i] = m.setdefault(x, len(m))
    return out


def load(d):
    z = np.load(f"{d}/invocation_fp16_requests.npz")
    r = json.load(open(f"{d}/result.json"))
    return z, r


def compare(a, b):
    za, ra = load(a)
    zb, rb = load(b)
    out = {"requests": (int((za["action"] == 1).sum()), int((zb["action"] == 1).sum()))}
    for k in ("action", "size", "load_events"):
        out[k] = bool(za[k].shape == zb[k].shape and (za[k] == zb[k]).all())
    out["pairing"] = bool(za["alloc_id"].shape == zb["alloc_id"].shape and
                          (canon_pairs(za["action"], za["alloc_id"]) == canon_pairs(zb["action"], zb["alloc_id"])).all())
    out["m4_local"] = int(za["m4_local"]) == int(zb["m4_local"])
    out["summary_equal"] = ra.get("summary") == rb.get("summary")
    out["threads"] = ((ra.get("environment") or {}).get("threads"), (rb.get("environment") or {}).get("threads"))
    out["identical"] = all(out[k] for k in ("action", "size", "load_events", "pairing", "m4_local"))
    if not out["identical"]:
        sa, sb = za["size"][za["action"] == 1], zb["size"][zb["action"] == 1]
        n = min(len(sa), len(sb))
        diff = np.nonzero(sa[:n] != sb[:n])[0]
        out["first_size_divergence_request"] = int(diff[0]) if len(diff) else (n if len(sa) != len(sb) else None)
        out["request_bytes"] = (int(sa.sum()), int(sb.sum()))
    return out


if __name__ == "__main__":
    ref = sys.argv[1]
    for other in sys.argv[2:]:
        print(json.dumps({"ref": ref, "other": other, **compare(ref, other)}))
