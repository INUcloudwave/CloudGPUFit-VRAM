"""Conv models (60 conditions), every measured GPU, with EACH GPU GENERATION'S OWN cuDNN table (2026-11-26).
Tables (probe v3, same script, 4,206 shapes, candidate 1): RTX 3090 (Ampere; also A10G), AWS g6 L4 (Ada; also gr6 L4,
g6e L40S), AWS g4dn T4 (Turing), AWS g7e RTX PRO 6000 (Blackwell; also g7 RTX PRO 4500, same sm_120, cu128 image).
Same variants as final_eval.py (unchanged): v3 / step1 / table / table + cuBLASLt + max_pool, with the 3090 table and
with the own-generation table. Measured values: AWS measure.json (first_measure_peak_reserved), 3090 L256 csv.
Conditions with allocator retries are left out (as final_eval.py). g7e has no gpu_profiles entry: capacity =
cuda_total_memory (no retries in these conditions), cuBLAS workspace from gpu_rules with its own name/capability/torch.
usage (image): python final_eval_pergpu.py <aws extract root> <3090 L256 csv> <probe dir> <out dir>"""
import csv, glob, json, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, "/exp/src")
from cgf680 import replay3, replay4
from cgf680.gpu_rules import cublas_workspace

E = Path("/exp")
ROOT, CSV3090, PDIR, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
M = 1 << 20
CNN = ("resnet-18", "resnet-50", "mobilenet-v2-1.0-224")
KEYS = ("N", "C", "H", "W", "K", "R", "stride", "pad", "dil", "groups")
TABLE_FILES = {"ampere_3090": PDIR / "probe_rtx3090_cudnn9.1.0.jsonl", "ada_g6": PDIR / "probe_aws-g6.xlarge.jsonl",
               "turing_g4dn": PDIR / "probe_aws-g4dn.xlarge.jsonl", "blackwell_g7e": PDIR / "g7e/cudnn_probe_g7e/probe_g7e.jsonl"}
GPUS = {  # label -> (aws dir or None, gpu_profiles key or None, own table)
    "rtx3090": (None, "rtx3090", "ampere_3090"),
    "g5.xlarge (A10G)": ("g5.xlarge-l256", "a10g", "ampere_3090"),
    "g6.xlarge (L4)": ("g6.xlarge-l256", "l4", "ada_g6"),
    "gr6.4xlarge (L4)": ("gr6.4xlarge-l256", "l4", "ada_g6"),
    "g6e.xlarge (L40S)": ("g6e.xlarge-l256", "l40s", "ada_g6"),
    "g4dn.xlarge (T4)": ("g4dn.xlarge-l256", "t4", "turing_g4dn"),
    "g7.2xlarge (RTX PRO 4500)": ("g7.2xlarge__l256__cu128", "rtxpro4500", "blackwell_g7e"),
    "g7e.2xlarge (RTX PRO 6000)": ("g7e.2xlarge__l256__cu128", None, "blackwell_g7e"),
}
TAB = {}
for name, f in TABLE_FILES.items():
    t = {}
    for line in open(f):
        r = json.loads(line)
        if r.get("status") == "ok":
            t[tuple(r[k] for k in KEYS)] = r["candidates"][0]["ws"]
    TAB[name] = t


def pkey(cp):
    n, c, h, w = cp["input"]
    pad = cp["padding"] if isinstance(cp["padding"], list) else [0, 0]
    return (n, c, h, w, cp["weight"][0], cp["weight"][2], cp["stride"][0], pad[0], cp["dilation"][0], cp["groups"])


def stream(lab, ws_rel, drop_conv=True, tab=None, lt=False, pool=False):
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
            if tab is not None and is_conv[i] and c in lab["conv"]:
                w = tab.get(pkey(lab["conv"][c]), 0)
            elif lt and func[i] == "linear":
                w = 1 << 20
            elif pool and func[i] == "max_pool2d":
                w = 4 * int(size[i])
            if w:
                A += [1, 0]; S += [w, w]; K += [nxt, nxt]; nxt -= 1
    return {"load": lab["load"], "act": A, "size": S, "aid": K, "m4_local": None}, wr


profiles = json.loads((E / "manifest/gpu_profiles.json").read_text())["profiles"]
pos = {r["case_id"]: r["ws_insert_rel"] for r in map(json.loads, open(E / "runs/gemm_exact/positions.jsonl"))
       if r.get("status") == "ok" and r.get("prefix_equal_to_npz")}
meas, ginfo = {}, {}
for r in csv.DictReader(open(CSV3090)):
    if r["condition_id"].split("__")[0] in CNN and r["measure_status"] == "ok":
        meas[("rtx3090", r["condition_id"])] = (int(r["m1_peak_reserved"]), False)
for g, (d, prof, own) in GPUS.items():
    if prof:
        p = profiles[prof]
        ginfo[g] = {"cap": p["capacity_bytes"], "ws": cublas_workspace(p["gpu_name"], p.get("compute_capability"), p.get("runtime_torch", "2.6.0"))["bytes"]}
    if d is None:
        continue
    for f in glob.glob(f"{ROOT}/{d}/runs/*/*/measure.json"):
        cid = os.path.basename(os.path.dirname(f))
        if cid.split("__")[0] not in CNN:
            continue
        m = json.load(open(f))
        if m["status"] != "ok":
            continue
        e = m["environment"]
        if g not in ginfo:
            ginfo[g] = {"cap": e["cuda_total_memory"], "ws": cublas_workspace(e["gpu_name"], e["capability"], e["torch"].split("+")[0])["bytes"]}
        meas[(g, cid)] = (m["summary"]["first_measure_peak_reserved"], m["measures"][0]["stats_end"]["num_alloc_retries"] > 0)
cases = sorted({c for _, c in meas})
VARS = ["v3", "step1", "table_3090", "table_3090+lt+pool", "table_own", "table_own+lt+pool"]


def run(c):
    lab = replay4.load_labels(E / f"runs/layer_labels/{c}.npz")
    out = {"case_id": c}
    for g, (_, _, own) in GPUS.items():
        if (g, c) not in meas:
            continue
        for v in VARS:
            kw = {"v3": dict(drop_conv=False), "step1": dict()}.get(v)
            if kw is None:
                t = TAB["ampere_3090"] if "3090" in v else TAB[own]
                kw = dict(tab=t, lt="+lt" in v, pool="+lt" in v)
            st, wr = stream(lab, pos.get(c), **kw)
            r = replay3.replay(st, ginfo[g]["cap"], "descending", wr, ginfo[g]["ws"])
            out[f"{g}|{v}"] = None if r["oom"] else r["m0_start_reserved"]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(8) as p:
        rows = p.map(run, cases)
    (OUT / "final_eval_pergpu_replays.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    rep = {"tables": {k: str(v) for k, v in TABLE_FILES.items()}, "gpu_info": ginfo, "result": {}}
    for g in GPUS:
        for v in VARS:
            errs = [(r[f"{g}|{v}"] - meas[(g, r["case_id"])][0], meas[(g, r["case_id"])][0]) for r in rows
                    if (g, r["case_id"]) in meas and not meas[(g, r["case_id"])][1] and r.get(f"{g}|{v}") is not None]
            if not errs:
                continue
            a = [abs(e) for e, _ in errs]
            rep["result"][f"{g}|{v}"] = {"n": len(errs), "within100MB": sum(x <= 1e8 for x in a), "exact": sum(x == 0 for x in a),
                                         "le2MiB": sum(x <= 2 * M for x in a), "max_over_MiB": round(max(e for e, _ in errs) / M, 1),
                                         "max_under_MiB": round(min(e for e, _ in errs) / M, 1)}
    (OUT / "final_eval_pergpu_summary.json").write_text(json.dumps(rep, indent=1))
    for g in GPUS:
        print(g, {v: (lambda s: f"{s['within100MB']}/{s['n']} ex{s['exact']} [{s['max_under_MiB']},{s['max_over_MiB']}]")(rep["result"][f"{g}|{v}"])
                  for v in VARS if f"{g}|{v}" in rep["result"]})


if __name__ == "__main__":
    main()
