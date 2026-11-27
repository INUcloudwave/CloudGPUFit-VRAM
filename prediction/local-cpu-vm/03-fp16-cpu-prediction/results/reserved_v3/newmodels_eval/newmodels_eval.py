"""Generalisation test (2026-11-23): 6 models never used for any rule, RTX 3090 measurement (INUGPU newmodels_20261122).
LLM : v3 prediction (tools/reserved_predict.py, results/reserved_v3/newmodels_rtx3090*) R_at effective vs m1_peak_reserved.
CNN : same variants as cudnn_fit/final_eval.py (v3 / step1 / cuDNN table / + cuBLASLt + max_pool), the table being
      (a) known : RTX 3090 probe v3 (3,253 shapes of older models + grid); shapes not in it -> 0
      (b) grid  : (a) + family/grid rule (cudnn_fit/grid_rule) for the missing shapes (missing family -> 0)
      (c) new   : (a) + RTX 3090 probe of the 953 new shapes (when available: argv[3])
usage (image): python newmodels_eval.py <gpu per_condition.csv> <out dir> [probe_newmodels.jsonl]"""
import csv, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, "/exp/src")
sys.path.insert(0, "/exp/results/reserved_v3/cudnn_fit/grid_rule")
from cgf680 import replay3, replay4
from cgf680.gpu_rules import cublas_workspace

E = Path("/exp")
PERCOND, OUT = sys.argv[1], Path(sys.argv[2])
NEWPROBE = sys.argv[3] if len(sys.argv) > 3 else None
M = 1 << 20
CNN = ("convnext-tiny-224", "efficientnet-b0", "regnet-y-040")
KEYS = ("N", "C", "H", "W", "K", "R", "stride", "pad", "dil", "groups")


def load_probe(f):
    d = {}
    for line in open(f):
        r = json.loads(line)
        if r.get("status") == "ok":
            d[tuple(r[k] for k in KEYS)] = r["candidates"][0]["ws"]
    return d


KNOWN = load_probe(E / "gpu_probe/results/rtx3090_v3/probe_rtx3090.jsonl")
GRID = {tuple(r[k] for k in KEYS): r["ws_pred"] for r in map(json.loads, open(E / "results/reserved_v3/cudnn_fit/grid_rule/pred_newmodels_rtx3090.jsonl"))
        if r["ws_pred"] is not None}
NEW = load_probe(NEWPROBE) if NEWPROBE else {}


def pkey(cp):
    n, c, h, w = cp["input"]
    k = cp["weight"][2]
    pad = cp["padding"]
    if pad == "valid":
        pad = [0, 0]
    elif pad == "same":
        pad = [(k - 1) // 2 * cp["dilation"][0]] * 2
    return (n, c, h, w, cp["weight"][0], k, cp["stride"][0], pad[0], cp["dilation"][0], cp["groups"])


def table(name):
    if name == "known":
        return KNOWN
    if name == "grid":
        return {**GRID, **KNOWN}
    return {**NEW, **KNOWN}


def stream(lab, ws_rel, drop_conv=True, tab=None, lt=False, pool=False, cov=None):
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
                key = pkey(lab["conv"][c])
                w = tab.get(key, 0)
                if cov is not None:
                    cov["calls"] += 1
                    cov["covered"] += key in tab
            elif lt and func[i] == "linear":
                w = 1 << 20
            elif pool and func[i] == "max_pool2d":
                w = 4 * int(size[i])
            if w:
                A += [1, 0]; S += [w, w]; K += [nxt, nxt]; nxt -= 1
    return {"load": lab["load"], "act": A, "size": S, "aid": K, "m4_local": None}, wr


TABLES = ["known", "grid"] + (["new"] if NEW else [])
VARIANTS = {"v3": dict(drop_conv=False), "step1": dict()}
for t in TABLES:
    VARIANTS[f"table_{t}"] = dict(tab=t)
    VARIANTS[f"table_{t}+lt+pool"] = dict(tab=t, lt=True, pool=True)
prof = json.loads((E / "manifest/gpu_profiles.json").read_text())["profiles"]["rtx3090"]
WSB = cublas_workspace(prof["gpu_name"], prof.get("compute_capability"), prof.get("runtime_torch", "2.6.0"))["bytes"]
pos = {r["case_id"]: r["ws_insert_rel"] for r in map(json.loads, open(E / "runs/gemm_exact_newmodels/positions.jsonl"))
       if r.get("status") == "ok" and r.get("prefix_equal_to_npz")}
meas = {r["condition_id"]: int(r["m1_peak_reserved"]) for r in csv.DictReader(open(PERCOND))}


def run(c):
    lab = replay4.load_labels(E / f"runs/layer_labels_newmodels/{c}.npz")
    out = {"case_id": c, "measured": meas[c]}
    for name, kw in VARIANTS.items():
        kw = dict(kw)
        cov = None
        if "tab" in kw:
            tname = kw["tab"]
            kw["tab"] = table(tname)
            cov = {"calls": 0, "covered": 0}
        st, wr = stream(lab, pos.get(c), cov=cov, **kw)
        r = replay3.replay(st, prof["capacity_bytes"], "descending", wr, WSB)
        out[name] = None if r["oom"] else r["m0_start_reserved"]
        if cov is not None and name.endswith(tname):
            out[f"coverage_{tname}"] = cov
    return out


def stats(errs):
    a = [abs(e) for e, _ in errs]
    return {"n": len(errs), "within100MB": sum(x <= 1e8 for x in a), "exact": sum(x == 0 for x in a),
            "le2MiB": sum(x <= 2 * M for x in a), "max_over_MiB": round(max(e for e, _ in errs) / M, 1),
            "max_under_MiB": round(min(e for e, _ in errs) / M, 1),
            "max_abs_pct": round(max(abs(e) / m * 100 for e, m in errs), 1)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = sorted(c for c in meas if c.split("__")[0] in CNN)
    with Pool(8) as p:
        rows = p.map(run, cases)
    (OUT / "cnn_replays.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    rep = {"gpu": "rtx3090", "new_probe": NEWPROBE, "cnn": {}, "llm": {}}
    for mdl in CNN + ("all_cnn",):
        rr = [r for r in rows if mdl == "all_cnn" or r["case_id"].startswith(mdl + "__")]
        rep["cnn"][mdl] = {v: stats([(r[v] - r["measured"], r["measured"]) for r in rr if r[v] is not None]) for v in VARIANTS}
    preds = {}
    for n in ("newmodels_rtx3090", "newmodels_rtx3090_retry2"):
        for line in open(E / f"results/reserved_v3/{n}/predictions.jsonl"):
            r = json.loads(line)
            if r.get("status") == "ok":
                preds[r["case_id"]] = r
    llm_rows = []
    for c, m in sorted(meas.items()):
        if c.split("__")[0] in CNN:
            continue
        p = preds.get(c)
        v = p["main"]["R_at"]["effective"]["reserved"] if p else None
        llm_rows.append({"case_id": c, "measured": m, "v3": v, "R_inf": p["main"]["R_inf"] if p else None})
    (OUT / "llm_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in llm_rows))
    for mdl in sorted({r["case_id"].split("__")[0] for r in llm_rows}) + ["all_llm"]:
        rr = [r for r in llm_rows if (mdl == "all_llm" or r["case_id"].startswith(mdl + "__")) and r["v3"] is not None]
        rep["llm"][mdl] = stats([(r["v3"] - r["measured"], r["measured"]) for r in rr])
        rep["llm"][mdl]["missing"] = sum(1 for r in llm_rows if (mdl == "all_llm" or r["case_id"].startswith(mdl + "__")) and r["v3"] is None)
    (OUT / "summary.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
