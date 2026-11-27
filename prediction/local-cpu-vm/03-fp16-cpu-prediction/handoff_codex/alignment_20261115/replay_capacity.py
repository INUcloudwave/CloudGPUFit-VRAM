"""Re-replay saved CPU request traces (invocation_fp16_requests.npz) with a GPU-like allocator capacity.
Analysis only: reads runs/main395/cases/<cid>/, writes JSON to stdout. Production code/results untouched."""
import json, sys, time
import numpy as np
import cgf680.cpu_worker2 as W
cap = int(sys.argv[1]); ids = sys.argv[2:]
W.CAP = cap
for cid in ids:
    t = time.time()
    z = np.load(f"/exp/runs/main395/cases/{cid}/invocation_fp16_requests.npz")
    seg = {k: z[k] for k in ("action", "size", "alloc_id")}
    load_events = [tuple(x) for x in z["load_events"].tolist()]
    al, runs, info = W.replay_protocol(load_events, seg, int(z["m4_local"]))
    m0 = runs[-1]
    print(json.dumps({"cid": cid, "cap": cap, "oom": bool(info["oom"]), "runs_done": [r["run"] for r in runs],
                      "m1_start_reserved": m0["start"]["reserved"], "m1_peak_reserved": m0["peak"]["reserved"],
                      "m1_peak_allocated": m0["peak"]["allocated"], "sec": round(time.time() - t, 1)}), flush=True)
