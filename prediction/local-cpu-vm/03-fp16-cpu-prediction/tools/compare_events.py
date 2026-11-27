#!/usr/bin/env python3
"""Exact request-trace equality between two case runs (action, size, pairing, region, markers, resident)."""
import json, sys
import numpy as np
def load(d):
    z = np.load(f"{d}/events.npz"); r = json.load(open(f"{d}/result.json")); return z, r
def compare(a, b):
    za, ra = load(a); zb, rb = load(b)
    out = {}
    for k in ("action", "size", "alloc_id", "region", "resident_sizes", "marker_pos"):
        out[k] = bool(za[k].shape == zb[k].shape and (za[k] == zb[k]).all())
    out["prediction_equal"] = ra.get("prediction") == rb.get("prediction")
    out["inputs_equal"] = ra.get("inputs") == rb.get("inputs")
    out["events"] = int(len(za["action"])), int(len(zb["action"]))
    out["identical"] = all(v for k, v in out.items() if k != "events")
    return out
if __name__ == "__main__":
    print(json.dumps(compare(sys.argv[1], sys.argv[2])))
