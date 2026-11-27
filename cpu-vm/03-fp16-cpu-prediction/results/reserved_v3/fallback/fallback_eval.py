"""replay5 (cuDNN plan fallback) checks, RTX 3090 tables (2026-11-23).
 A) full capacity: replay5 == replay3 with candidate-1 workspaces (+cuBLASLt +max_pool), all 108 CNN conditions
 B) memory pressure: smallest capacity (2 MiB steps) without OOM, with vs without fallback, and the reserved value
    at a few capacities between them, for the conditions with the largest workspaces.
usage (image): python fallback_eval.py <out dir>"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, "/exp/src")
from cgf680 import replay3, replay4, replay5
from cgf680.gpu_rules import cublas_workspace

E = Path("/exp"); OUT = Path(sys.argv[1]); M = 1 << 20
KEYS = ("N", "C", "H", "W", "K", "R", "stride", "pad", "dil", "groups")
CANDS, END = {}, {}
for f in ("gpu_probe/results/rtx3090_v3/probe_rtx3090.jsonl", "gpu_probe/results/rtx3090_newmodels/probe_rtx3090_newmodels.jsonl"):
    for line in open(E / f):
        r = json.loads(line)
        if r.get("status") == "ok":
            k = tuple(r[x] for x in KEYS)
            CANDS[k] = [c["ws"] for c in r["candidates"] if not c.get("oom")]
            END[k] = r["end"]
prof = json.loads((E / "manifest/gpu_profiles.json").read_text())["profiles"]["rtx3090"]
WSB = cublas_workspace(prof["gpu_name"], prof.get("compute_capability"), prof.get("runtime_torch", "2.6.0"))["bytes"]
CAP = prof["capacity_bytes"]
pos = {}
for f in ("runs/gemm_exact/positions.jsonl", "runs/gemm_exact_newmodels/positions.jsonl"):
    for r in map(json.loads, open(E / f)):
        if r.get("status") == "ok" and r.get("prefix_equal_to_npz"):
            pos[r["case_id"]] = r["ws_insert_rel"]


def pkey(cp):
    n, c, h, w = cp["input"]; k = cp["weight"][2]; pad = cp["padding"]
    pad = [0, 0] if pad == "valid" else ([(k - 1) // 2 * cp["dilation"][0]] * 2 if pad == "same" else pad)
    return (n, c, h, w, cp["weight"][0], k, cp["stride"][0], pad[0], cp["dilation"][0], cp["groups"])


def streams(lab, ws_rel):
    """-> (replay3 stream with candidate-1 workspaces, replay5 stream with action-2 entries, cand table)"""
    act, size, aid, layer, func, call = lab["act"], lab["size"], lab["aid"], lab["layer"], lab["func"], lab["call"]
    is_conv = np.isin(func, list(replay4.CONV_FUNCS))
    drop = {int(aid[i]) for i in range(len(act)) if act[i] == 1 and layer[i] == 1 and (is_conv[i] or func[i] == "max_pool2d")}
    s3, s5, keys, seen, nxt = ([], [], []), ([], [], []), [], set(), -1
    w3 = w5 = None
    for i in range(len(act)):
        if ws_rel is not None and i == ws_rel:
            w3, w5 = len(s3[0]), len(s5[0])
        if int(aid[i]) in drop:
            continue
        for s in (s3, s5):
            s[0].append(int(act[i])); s[1].append(int(size[i])); s[2].append(int(aid[i]))
        c = int(call[i])
        if act[i] == 1 and layer[i] == 0 and c not in seen and (is_conv[i] or func[i] in ("linear", "max_pool2d")):
            seen.add(c)
            if is_conv[i] and c in lab["conv"]:
                k = pkey(lab["conv"][c])
                cl = CANDS.get(k, [0])
                if cl[0]:
                    s3[0].extend([1, 0]); s3[1].extend([cl[0]] * 2); s3[2].extend([nxt] * 2); nxt -= 1
                if k not in keys:
                    keys.append(k)
                s5[0].append(2); s5[1].append(keys.index(k)); s5[2].append(0)
            else:
                w = (1 << 20) if func[i] == "linear" else 4 * int(size[i])
                for s in (s3, s5):
                    s[0].extend([1, 0]); s[1].extend([w] * 2); s[2].extend([nxt] * 2)
                nxt -= 1
    mk = lambda s: {"load": lab["load"], "act": s[0], "size": s[1], "aid": s[2], "m4_local": None}
    return mk(s3), w3, mk(s5), w5, [CANDS.get(k, [0]) for k in keys], keys


def lab_of(c):
    p = E / f"runs/layer_labels/{c}.npz"
    return replay4.load_labels(p if p.exists() else E / f"runs/layer_labels_newmodels/{c}.npz")


def check_full(c):
    s3, w3, s5, w5, cl, _ = streams(lab_of(c), pos.get(c))
    a = replay3.replay(s3, CAP, "descending", w3, WSB)
    b = replay5.replay(s5, CAP, cl, "descending", w5, WSB)
    return {"case_id": c, "r3": a.get("m0_start_reserved"), "r5": b.get("m0_start_reserved"),
            "r3_peak": a.get("m0_peak_reserved"), "r5_peak": b.get("m0_peak_reserved"), "fallbacks": b["fallback"]["fallbacks"]}


def min_cap(fn, lo, hi):
    lo, hi = lo // (2 * M), -(-hi // (2 * M))
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fn(mid * 2 * M)["oom"]:
            lo = mid
        else:
            hi = mid
    return hi * 2 * M


def pressure(c):
    s3, w3, s5, w5, cl, keys = streams(lab_of(c), pos.get(c))
    f3 = lambda cap: replay3.replay(s3, cap, "descending", w3, WSB)
    f5 = lambda cap: replay5.replay(s5, cap, cl, "descending", w5, WSB)
    full = f3(CAP)
    hi = full["max_reserved"] + 2 * M
    m3 = min_cap(f3, 0, hi)
    m5 = min_cap(f5, 0, hi)
    pts = []
    for frac in (1.0, 0.75, 0.5, 0.25, 0.0):
        cap = int(m5 + frac * (m3 - m5)) // (2 * M) * (2 * M)
        r = f5(cap)
        pts.append({"cap_MiB": cap // M, "oom": bool(r["oom"]), "m0_peak_reserved_MiB": None if r["oom"] else r["m0_peak_reserved"] // M,
                    "fallbacks": r["fallback"]["fallbacks"], "keys_fallen_back": len(r["fallback"]["keys"])})
    return {"case_id": c, "R_full_MiB": full["m0_peak_reserved"] // M, "min_cap_no_fallback_MiB": m3 // M,
            "min_cap_with_fallback_MiB": m5 // M, "points": pts}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = sorted({p.stem for p in (E / "runs/layer_labels").glob("*.npz")} | {p.stem for p in (E / "runs/layer_labels_newmodels").glob("*.npz")})
    cases = [c for c in cases if not c.startswith("vit")]
    with Pool(8) as p:
        a = p.map(check_full, cases)
    same = [r for r in a if r["r3"] == r["r5"] and r["r3_peak"] == r["r5_peak"]]
    print("A) full capacity: replay5 == replay3 for", len(same), "/", len(a), "; fallbacks at full cap:", sum(r["fallbacks"] for r in a), flush=True)
    big = ["resnet-50__b128__r512", "resnet-18__b128__r512", "regnet-y-040__b128__r512", "convnext-tiny-224__b128__r512", "mobilenet-v2-1.0-224__b128__r512"]
    with Pool(5) as p:
        b = p.map(pressure, [c for c in big if c in cases])
    for r in b:
        print("B)", json.dumps(r), flush=True)
    (OUT / "fallback_eval.json").write_text(json.dumps({"full": a, "pressure": b}, indent=1))


if __name__ == "__main__":
    main()
