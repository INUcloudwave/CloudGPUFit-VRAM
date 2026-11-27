"""Remaining errors after the RTX 3090 probe table: test GPU library requests that are not conv workspaces, seen in
the 3090 traces (wsmap): cuBLASLt 1 MiB transient after the classifier `linear`, and max_pool2d indices (transient
4 x output, int64) instead of the CPU max_pool2d library buffer. usage (image): python extra_lib_test.py <per_condition.csv>"""
import csv, json, sys
sys.path.insert(0, "/exp/src")
import numpy as np
from cgf680 import replay3, replay4
from cgf680.gpu_rules import cublas_workspace
E = "/exp"
PROBE = {}
for line in open(f"{E}/gpu_probe/results/rtx3090/probe_rtx3090.jsonl"):
    r = json.loads(line)
    if r.get("status") == "ok":
        PROBE[(r["N"], r["C"], r["H"], r["W"], r["K"], r["R"], r["stride"], r["pad"], r["dil"], r["groups"])] = r["candidates"][0]["ws"]


def pkey(cp):
    n, c, h, w = cp["input"]
    pad = cp["padding"] if isinstance(cp["padding"], list) else [0, 0]
    return (n, c, h, w, cp["weight"][0], cp["weight"][2], cp["stride"][0], pad[0], cp["dilation"][0], cp["groups"])


pos = {}
for line in open(f"{E}/runs/gemm_exact/positions.jsonl"):
    r = json.loads(line)
    if r.get("status") == "ok" and r.get("prefix_equal_to_npz"):
        pos[r["case_id"]] = r["ws_insert_rel"]
meas = {r["case_id"]: int(r["rtx3090_res"]) for r in csv.DictReader(open(sys.argv[1]))}
M = 1 << 20


def stream(lab, ws_rel, lt=False, pool=False):
    act, size, aid, layer, func = lab["act"], lab["size"], lab["aid"], lab["layer"], lab["func"]
    call = lab["call"]
    is_conv = np.isin(func, list(replay4.CONV_FUNCS))
    drop = {int(aid[i]) for i in range(len(act)) if act[i] == 1 and layer[i] == 1 and (is_conv[i] or (pool and func[i] == "max_pool2d"))}
    A, S, K, nxt, seen, wr = [], [], [], -1, set(), None
    for i in range(len(act)):
        if ws_rel is not None and i == ws_rel:
            wr = len(A)
        if int(aid[i]) in drop:
            continue
        A.append(int(act[i])); S.append(int(size[i])); K.append(int(aid[i]))
        c = int(call[i])
        if act[i] == 1 and layer[i] == 0 and c not in seen:
            w = 0
            if is_conv[i] and c in lab["conv"]:
                w = PROBE.get(pkey(lab["conv"][c]), 0)
            elif lt and func[i] == "linear":
                w = 1 << 20
            elif pool and func[i] == "max_pool2d":
                w = 4 * int(size[i])
            if w or is_conv[i] or func[i] in ("linear", "max_pool2d"):
                seen.add(c)
            if w:
                A += [1, 0]; S += [w, w]; K += [nxt, nxt]; nxt -= 1
    return {"load": lab["load"], "act": A, "size": S, "aid": K, "m4_local": None}, wr


for c in sorted(meas):
    if c.split("__")[0] not in ("resnet-18", "resnet-50", "mobilenet-v2-1.0-224"):
        continue
    lab = replay4.load_labels(f"{E}/runs/layer_labels/{c}.npz")
    out = {"case_id": c, "meas_MiB": meas[c] / M}
    for name, kw in (("table", {}), ("table+cublasLt", {"lt": True}), ("table+cublasLt+pool_idx", {"lt": True, "pool": True})):
        st, wr = stream(lab, pos.get(c), **kw)
        r = replay3.replay(st, 24979374080, "descending", wr, 8519680)
        out[name] = round(r["m0_start_reserved"] / M - meas[c] / M, 1)
    print(json.dumps(out), flush=True)
