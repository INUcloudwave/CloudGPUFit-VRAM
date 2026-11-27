"""Pre-registered predictions at given allocator caps (MiB): replay3 (candidate 1 only) vs replay5 (fallback).
usage (image): python cap_predict.py <case_id> <cap MiB>... -> JSON lines"""
import json, sys
cid, caps = sys.argv[1], [int(x) for x in sys.argv[2:]]
sys.argv = [sys.argv[0], "/tmp/unused"]
sys.path.insert(0, "/exp/results/reserved_v3/fallback")
import fallback_eval as fe
from cgf680 import replay3, replay5
M = 1 << 20
s3, w3, s5, w5, cl, keys = fe.streams(fe.lab_of(cid), fe.pos.get(cid))
for cap in caps:
    a = replay3.replay(s3, cap * M, "descending", w3, fe.WSB)
    b = replay5.replay(s5, cap * M, cl, "descending", w5, fe.WSB)
    print(json.dumps({"case_id": cid, "cap_MiB": cap,
                      "replay3": "oom" if a["oom"] else {"reserved": a["m0_peak_reserved"] // M, "allocated": round(a["m0_peak_allocated"] / M, 1)},
                      "replay5": "oom" if b["oom"] else {"reserved": b["m0_peak_reserved"] // M, "allocated": round(b["m0_peak_allocated"] / M, 1),
                                                         "fallback_keys": len(b["fallback"]["keys"])}}), flush=True)
