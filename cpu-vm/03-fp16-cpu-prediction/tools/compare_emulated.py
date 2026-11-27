#!/usr/bin/env python3
"""Compare an FP32-emulated FP16 request trace with a true FP16 CPU trace (sequence and predictions)."""
import json, sys
import numpy as np
emu_dir, ref_dir = sys.argv[1], sys.argv[2]
e = np.load(f"{emu_dir}/events_fp16_emulated.npz"); r = np.load(f"{ref_dir}/events.npz")
re_, rr = json.load(open(f"{emu_dir}/result.json")), json.load(open(f"{ref_dir}/result.json"))
ea = e["size"][e["action"] == 1]; ra = r["size"][r["action"] == 1]
out = {"emu_requests": int(len(ea)), "ref_requests": int(len(ra)),
       "request_sizes_identical_sequence": bool(len(ea) == len(ra) and (ea == ra).all()),
       "event_sequence_identical": bool(len(e["action"]) == len(r["action"]) and (e["action"] == r["action"]).all()
                                        and (e["size"] == r["size"]).all()),
       "emu_request_bytes": int(ea.sum()), "ref_request_bytes": int(ra.sum())}
if len(ea) == len(ra):
    d = np.nonzero(ea != ra)[0]
    out["n_size_diff"] = int(len(d)); out["first_diffs"] = [(int(i), int(ea[i]), int(ra[i])) for i in d[:8]]
pe, pr = re_["prediction"], rr["prediction"]
out["pred"] = {k: [pe[k], pr[k], pe[k] - pr[k], round((pe[k] - pr[k]) / pr[k] * 100, 3) if pr[k] else None]
               for k in pe if k.endswith("_bytes") and pe.get(k) is not None and pr.get(k) is not None}
out["fp32_raw_peak"] = {k: re_["prediction_fp32_raw"][k] for k in ("allocated_peak_bytes", "reserved_peak_bytes")}
out["emulation"] = {k: re_["fp16_emulation"][k] for k in ("emulated_casts", "early_freed_downcast_sources")}
out["match"] = {k: re_["fp16_emulation"]["match"][k] for k in ("records", "matched", "unmatched_records", "tag_counts")}
print(json.dumps(out, indent=None))
