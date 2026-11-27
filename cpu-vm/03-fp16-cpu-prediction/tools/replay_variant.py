#!/usr/bin/env python3
"""Replay a recorded request trace with a size transform (e.g. FP32 trace halved) and compare with a
reference prediction (e.g. true FP16 CPU trace).

  docs/run_in_image.sh tools/replay_variant.py --events runs/fp32-calib/cases/<fp32 id> \
      --resident-from runs/pilot/cases/<fp16 id> --scale 0.5 --ref runs/pilot/cases/<fp16 id>

--scale applies to every traced request (ceil(size*scale)); resident sizes are taken verbatim from
--resident-from (actual FP16 parameter/buffer/input storages), so only precision-affected request
sizes change. Integer tensors in the trace are scaled too (they cannot be told apart in the profiler
stream); they are small (token ids, masks) — the error this causes is part of what is measured.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from cgf680.cpu_worker import replay, measure_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--resident-from", required=True)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--ref", default=None)
    a = ap.parse_args()
    z = np.load(Path(a.events) / "events.npz")
    zr = np.load(Path(a.resident_from) / "events.npz")
    ev = {k: z[k] for k in ("action", "size", "alloc_id", "region")}
    ev["size"] = np.ceil(ev["size"] * a.scale).astype(np.int64)
    mp = {k: int(v) for k, v in enumerate(z["marker_pos"])}
    names = [f"r{i}" for i in range(int(ev["region"].max()) + 1)]
    rp, snap, seg = replay(ev, mp, [int(x) for x in zr["resident_sizes"]], names)
    pred = measure_stats(snap, seg)
    out = {"events": a.events, "scale": a.scale, "prediction": pred}
    if a.ref:
        ref = json.loads((Path(a.ref) / "result.json").read_text())["prediction"]
        out["ref"] = a.ref
        out["diff"] = {k: {"variant": pred[k], "ref": ref[k], "diff_bytes": pred[k] - ref[k],
                           "rel": (pred[k] - ref[k]) / ref[k] if ref[k] else None}
                       for k in pred if k.endswith("_bytes") and pred[k] is not None and ref.get(k) is not None}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
