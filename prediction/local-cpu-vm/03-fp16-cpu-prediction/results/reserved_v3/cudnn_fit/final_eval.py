"""Vision models, every measured GPU: v3 / step 1 (drop CPU conv library) / + RTX 3090 cuDNN workspace table (probe v3,
candidate 1 per exact conv signature) / + two non-cuDNN GPU library requests seen in RTX 3090 traces:
cuBLASLt workspace 1 MiB (transient, after the classifier `linear` output) and max_pool2d int64 indices (transient,
4 x output; the CPU max_pool2d library buffer is dropped instead). R_at with each GPU's effective capacity.
usage (image): python final_eval.py <per_condition.csv> <probe jsonl> <out dir>"""
import csv, json, statistics, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, "/exp/src")
from cgf680 import replay3, replay4
from cgf680.gpu_rules import cublas_workspace

E = Path("/exp")
PERCOND, PROBEF, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
M = 1 << 20
CNN = ("resnet-18", "resnet-50", "mobilenet-v2-1.0-224")
GPU_PROFILE = {"rtx3090": "rtx3090", "g4dn.xlarge-l256": "t4", "g5.xlarge-l256": "a10g", "g6.xlarge-l256": "l4",
               "gr6.4xlarge-l256": "l4", "g6e.xlarge-l256": "l40s", "g7.2xlarge__l256__cu128": "rtxpro4500"}
PROBE = {}
for line in open(PROBEF):
    r = json.loads(line)
    if r.get("status") == "ok":
        PROBE[(r["N"], r["C"], r["H"], r["W"], r["K"], r["R"], r["stride"], r["pad"], r["dil"], r["groups"])] = r["candidates"][0]["ws"]


def pkey(cp):
    n, c, h, w = cp["input"]
    pad = cp["padding"] if isinstance(cp["padding"], list) else [0, 0]
    return (n, c, h, w, cp["weight"][0], cp["weight"][2], cp["stride"][0], pad[0], cp["dilation"][0], cp["groups"])


def stream(lab, ws_rel, drop_conv=True, conv_ws=False, lt=False, pool=False):
    act, size, aid, layer, func, call = lab["act"], lab["size"], lab["aid"], lab["layer"], lab["func"], lab["call"]
    is_conv = np.isin(func, list(replay4.CONV_FUNCS))
    drop = {int(aid[i]) for i in range(len(act)) if act[i] == 1 and layer[i] == 1
            and ((drop_conv and is_conv[i]) or (pool and func[i] == "max_pool2d"))}
    A, S, K, nxt, seen, wr = [], [], [], -1, set(), None
    for i in range(len(act)):
        if ws_rel is not None and i == ws_rel:
            wr = len(A)
        if int(aid[i]) in drop:
            continue
        A.append(int(act[i])); S.append(int(size[i])); K.append(int(aid[i]))
        c = int(call[i])
        if act[i] == 1 and layer[i] == 0 and c not in seen and (is_conv[i] or func[i] in ("linear", "max_pool2d")):
            seen.add(c)
            w = 0
            if conv_ws and is_conv[i] and c in lab["conv"]:
                w = PROBE.get(pkey(lab["conv"][c]), 0)
            elif lt and func[i] == "linear":
                w = 1 << 20
            elif pool and func[i] == "max_pool2d":
                w = 4 * int(size[i])
            if w:
                A += [1, 0]; S += [w, w]; K += [nxt, nxt]; nxt -= 1
    return {"load": lab["load"], "act": A, "size": S, "aid": K, "m4_local": None}, wr


VARIANTS = {"v3": dict(drop_conv=False), "step1": dict(), "cudnn_table": dict(conv_ws=True),
            "table+lt+pool": dict(conv_ws=True, lt=True, pool=True)}
profiles = json.loads((E / "manifest/gpu_profiles.json").read_text())["profiles"]
pos = {}
for line in (E / "runs/gemm_exact/positions.jsonl").read_text().splitlines():
    r = json.loads(line)
    if r.get("status") == "ok" and r.get("prefix_equal_to_npz"):
        pos[r["case_id"]] = r["ws_insert_rel"]
meas = {}
for r in csv.DictReader(open(PERCOND)):
    if r["model"] in CNN:
        meas[(r["gpu"], r["case_id"])] = (int(r["aws_res"]), r["retry"] == "True")
        meas[("rtx3090", r["case_id"])] = (int(r["rtx3090_res"]), False)
cases = sorted({c for _, c in meas})
gpus = sorted({g for g, _ in meas})


def run(c):
    lab = replay4.load_labels(E / f"runs/layer_labels/{c}.npz")
    out = {"case_id": c}
    for g in gpus:
        prof = profiles[GPU_PROFILE[g]]
        wsb = cublas_workspace(prof["gpu_name"], prof.get("compute_capability"), prof.get("runtime_torch", "2.6.0"))["bytes"]
        for name, kw in VARIANTS.items():
            st, wr = stream(lab, pos.get(c), **kw)
            r = replay3.replay(st, prof["capacity_bytes"], "descending", wr, wsb)
            out[f"{g}|{name}"] = None if r["oom"] else r["m0_start_reserved"]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(8) as p:
        rows = p.map(run, cases)
    (OUT / "final_eval_replays.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    rep = {"probe": PROBEF, "result": {}}
    for g in gpus:
        for name in VARIANTS:
            errs = []
            for r in rows:
                m = meas.get((g, r["case_id"])); v = r[f"{g}|{name}"]
                if m and not m[1] and v is not None:
                    errs.append((v - m[0], m[0]))
            a = [abs(e) for e, _ in errs]
            rep["result"][f"{g}|{name}"] = {
                "n": len(errs), "within100MB": sum(x <= 1e8 for x in a), "exact": sum(x == 0 for x in a),
                "le2MiB": sum(x <= 2 * M for x in a), "max_over_MiB": round(max(e for e, _ in errs) / M, 1),
                "max_under_MiB": round(min(e for e, _ in errs) / M, 1),
                "max_abs_pct": round(max(abs(e) / m * 100 for e, m in errs), 1)}
    (OUT / "final_eval_summary.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
