#!/usr/bin/env python3
"""cuDNN deviceless query (2026-11-23): for each conv signature, build the FP16 conv_fprop graph from SERIALIZED device
properties only (no handle, no GPU needed), ask the heuristics (mode A = INSTANT, then FALLBACK, the modes PyTorch 2.6
uses with benchmark=False) and record EVERY candidate plan in order: name, built ok, workspace bytes.
Two lists per signature: 'all' (no filter) and 'pt' (DOWN_CONVERT_INPUTS deselected, as PyTorch filters it).
PyTorch's choice is checked offline against the GPU probe (candidate 1 workspace).
usage: python deviceless_query.py --props props.json --signatures sigs.json --out out.jsonl [--limit N] [--save-props-from-gpu]
needs cudnn frontend (python) + libcudnn >= 9.11; on a CPU-only host also a libcuda stub on LD_LIBRARY_PATH."""
import argparse
import json
import os
import platform
import time

import cudnn

KEYS = ("N", "C", "H", "W", "K", "R", "stride", "pad", "dil", "groups")
HALF, FLOAT = cudnn.data_type.HALF, cudnn.data_type.FLOAT


def plans(dp, sg, deselect, check=True):
    N, C, H, W, K, R = (sg[k] for k in ("N", "C", "H", "W", "K", "R"))
    s, p, d, g = sg["stride"], sg["pad"], sg["dil"], sg["groups"]
    gr = cudnn.pygraph(io_data_type=HALF, intermediate_data_type=FLOAT, compute_data_type=FLOAT, device_property=dp)
    X = gr.tensor(name="X", dim=[N, C, H, W], stride=[C * H * W, H * W, W, 1], data_type=HALF)
    Wt = gr.tensor(name="W", dim=[K, C // g, R, R], stride=[(C // g) * R * R, R * R, R, 1], data_type=HALF)
    Y = gr.conv_fprop(image=X, weight=Wt, padding=[p, p], stride=[s, s], dilation=[d, d], compute_data_type=FLOAT)
    Y.set_output(True).set_data_type(HALF)
    gr.validate()
    gr.build_operation_graph()
    gr.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    if deselect:
        gr.deselect_numeric_notes([cudnn.numerical_note.DOWN_CONVERT_INPUTS])
    if check:   # 2026-11-23: check_support calls cudaGetDevice -> fails on a CPU-only host (driver stub); --no-check-support
        try:
            gr.check_support()
        except Exception as e:  # noqa: BLE001
            return {"error": f"check_support: {type(e).__name__}: {str(e)[:160]}"}
    out = []
    for i in range(gr.get_execution_plan_count()):
        c = {"i": i}
        try:
            c["name"] = gr.get_plan_name_at_index(i)
        except Exception:  # noqa: BLE001
            pass
        try:
            gr.build_plan_at_index(i)
            c["built"] = True
            c["ws"] = int(gr.get_workspace_size_plan_at_index(i))
        except Exception as e:  # noqa: BLE001
            c["built"] = False
            c["err"] = str(e)[:100]
        out.append(c)
    return {"cands": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--props", required=True)
    ap.add_argument("--signatures")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-check-support", action="store_true", help="CPU-only host: skip check_support (needs a device)")
    ap.add_argument("--save-props-from-gpu", action="store_true", help="GPU host only: write --props from device 0 and exit")
    a = ap.parse_args()
    if a.save_props_from_gpu:
        open(a.props, "w").write(cudnn.create_device_properties(0).serialize())
        print("PROPS_SAVED", a.props, os.path.getsize(a.props))
        return
    dp = cudnn.create_device_properties(open(a.props).read())
    sigs = json.load(open(a.signatures))["signatures"][: a.limit]
    t0 = time.time()
    with open(a.out, "w") as f:
        f.write(json.dumps({"environment": {"cudnn_backend": cudnn.backend_version(), "frontend": cudnn.__version__,
                                            "props_file": os.path.basename(a.props), "host": platform.node()[:0] or "masked",
                                            "python": platform.python_version()}}) + "\n")
        for n, sg in enumerate(sigs, 1):
            r = {k: sg[k] for k in KEYS}
            r["src"] = sg.get("src")
            for key, des in (("pt", True), ("all", False)):
                try:
                    r[key] = plans(dp, sg, des, not a.no_check_support)
                except Exception as e:  # noqa: BLE001
                    r[key] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            f.write(json.dumps(r) + "\n")
            if n % 200 == 0:
                f.flush()
                print(f"{n} signatures, {time.time() - t0:.0f} s", flush=True)
    print(f"DEVICELESS_DONE {len(sigs)} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
