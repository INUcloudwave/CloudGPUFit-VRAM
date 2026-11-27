"""Reserved prediction v4 candidates (2026-11-20): v3 replay (cgf680.replay3, unchanged) on a modified CPU stream.

Two optional modifications, both from CPU information only (labels from cgf680.layer_shadow):
  1. drop the CPU library layer: requests that are not outputs of a torch-function call (CPU kernel scratch such as
     oneDNN/mkldnn buffers). drop="conv": only those made inside conv calls (default; LLM/T5 internal temporaries that
     also exist on the GPU are kept), "all": every library request, "none": keep everything (= v3).
  2. GPU convolution workspace model: after the output request of every conv call, a transient request
     (allocated, then freed right away = the cuDNN workspace pattern seen in RTX 3090 traces) of size
     ws_fn(conv_params, output_bytes) is inserted. ws_fn=None: no conv workspace (= what the CPU alone knows).
The cuBLAS workspace position (ws_rel, index into the stored stream) is remapped onto the modified stream.
"""
from __future__ import annotations

import json

import numpy as np

from cgf680 import replay3

CONV_FUNCS = {"conv2d", "conv1d", "conv3d", "_convolution", "convolution"}


def load_labels(npz_path):
    z = np.load(npz_path)
    if not bool(z["equal_to_stored"]):
        raise ValueError(f"labels do not match the stored stream: {npz_path}")
    names = [str(x) for x in z["func_names"]]
    return {"act": z["action"].astype(np.int64), "size": z["size"].astype(np.int64),
            "aid": z["alloc_id"].astype(np.int64), "layer": z["layer"].astype(np.int64),
            "func": np.array([names[i] for i in z["func"]]), "call": z["call"].astype(np.int64),
            "load": [(int(a), int(s), int(k)) for a, s, k in z["load_events"].tolist()],
            "conv": {int(k): v for k, v in json.loads(str(z["conv_json"])).items()}}


def build_stream(lab, drop="conv", ws_fn=None, ws_rel=None):
    """-> (st for replay3.replay, remapped ws_rel, info)"""
    act, size, aid, layer, func, call = lab["act"], lab["size"], lab["aid"], lab["layer"], lab["func"], lab["call"]
    is_conv = np.isin(func, list(CONV_FUNCS))
    dropped = set()
    for i in range(len(act)):
        if act[i] == 1 and layer[i] == 1 and (drop == "all" or (drop == "conv" and is_conv[i])):
            dropped.add(int(aid[i]))
    out_a, out_s, out_k = [], [], []
    new_ws_rel = None
    seen_conv_out = set()
    ws_added, n_ws = 0, 0
    nxt = -1
    for i in range(len(act)):
        if ws_rel is not None and i == ws_rel:
            new_ws_rel = len(out_a)
        if int(aid[i]) in dropped:
            continue
        out_a.append(int(act[i]))
        out_s.append(int(size[i]))
        out_k.append(int(aid[i]))
        c = int(call[i])
        if (ws_fn is not None and act[i] == 1 and layer[i] == 0 and is_conv[i] and c in lab["conv"]
                and c not in seen_conv_out):
            seen_conv_out.add(c)
            w = int(ws_fn(lab["conv"][c], int(size[i])))
            if w > 0:
                out_a += [1, 0]
                out_s += [w, w]
                out_k += [nxt, nxt]
                nxt -= 1
                ws_added += w
                n_ws += 1
    if ws_rel is not None and new_ws_rel is None:
        new_ws_rel = len(out_a)
    st = {"load": lab["load"], "act": out_a, "size": out_s, "aid": out_k, "m4_local": None}
    info = {"drop": drop, "dropped_allocs": len(dropped),
            "dropped_bytes": int(sum(int(size[i]) for i in range(len(act)) if act[i] == 1 and int(aid[i]) in dropped)),
            "conv_ws_inserted": n_ws, "conv_ws_bytes_total": ws_added}
    return st, new_ws_rel, info


def predict(lab, caps, drop="conv", ws_fn=None, ws_rel=None, ws_bytes=replay3.CUBLAS_WS, policy="descending"):
    st, wr, info = build_stream(lab, drop, ws_fn, ws_rel)
    inf = replay3.replay(st, replay3.CAP_INF, policy, wr, ws_bytes)
    out = {"R_inf": inf["m0_start_reserved"], "max_reserved": inf["max_reserved"], "max_allocated": inf["max_allocated"],
           "ws_rel": wr, **info, "R_at": {}}
    for name, cap in caps.items():
        if inf["max_reserved"] <= cap:
            out["R_at"][name] = {"reserved": inf["m0_start_reserved"], "oom": False, "retry": False}
            continue
        r = replay3.replay(st, cap, policy, wr, ws_bytes)
        out["R_at"][name] = ({"reserved": None, "oom": True} if r["oom"] else
                             {"reserved": r["m0_start_reserved"], "oom": False, "retry": r["retry_released"] > 0})
    return out
