import numpy as np, sys
for spec in sys.argv[1:]:
    cid, d, pos = spec.split("=")
    z = np.load(d + "/invocation_fp16_requests.npz")
    print(cid, "load_events", len(z["load_events"]), "gpu-matched rel", int(pos) - len(z["load_events"]))
