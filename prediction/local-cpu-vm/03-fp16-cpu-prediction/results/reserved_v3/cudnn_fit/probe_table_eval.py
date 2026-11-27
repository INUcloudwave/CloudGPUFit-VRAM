"""Step 3 check: cuDNN workspace table from the RTX 3090 probe (gpu_probe/results/rtx3090, candidate 1 per exact conv
signature; no model, no trace) inserted after every conv output; compared with every measured GPU.
(Derived from table_eval.py, which used a ratio table from two RTX 3090 traces.)

Table (Ampere sm86, cuDNN 9.1.0, torch 2.6, FP16, benchmark=False): ws/output ratio per conv signature
(kernel, stride, groups, C_in, C_out), from ws_3090_traces.jsonl (b128 r512 traces of resnet-18/50, mobilenet).
Signatures not in the table: ws = 0 (= step 1). Then every vision condition is replayed (drop conv library + table
workspaces) and compared with each measured GPU. The table comes from 2 resnet conditions (b128 r512); the test is
whether it carries over to other batch sizes / resolutions (same arch: RTX 3090, A10G) and how it fares on other
architectures (expected to differ).
usage (image): python table_eval.py <per_condition.csv> <out dir>
"""
import csv
import json
import statistics
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, "/exp/src")
from cgf680 import replay4  # noqa: E402
from cgf680.gpu_rules import cublas_workspace  # noqa: E402

E = Path("/exp")
PERCOND, OUT = sys.argv[1], Path(sys.argv[2])
MIB = 1 << 20
CNN = ("resnet-18", "resnet-50", "mobilenet-v2-1.0-224")
GPU_PROFILE = {"rtx3090": "rtx3090", "g4dn.xlarge-l256": "t4", "g5.xlarge-l256": "a10g", "g6.xlarge-l256": "l4",
               "gr6.4xlarge-l256": "l4", "g6e.xlarge-l256": "l40s", "g7.2xlarge__l256__cu128": "rtxpro4500"}


def sig(cp):
    return (cp["weight"][2], cp["stride"][0], cp["groups"], cp["input"][1], cp["weight"][0])


PROBE = {}
for line in (E / "gpu_probe/results/rtx3090/probe_rtx3090.jsonl").read_text().splitlines():
    r = json.loads(line)
    if r.get("status") == "ok":
        PROBE[(r["N"], r["C"], r["H"], r["W"], r["K"], r["R"], r["stride"], r["pad"], r["dil"], r["groups"])] = r["candidates"][0]["ws"]
MISSING = set()


def pkey(cp):
    n, c, h, w = cp["input"]
    pad = cp["padding"] if isinstance(cp["padding"], list) else [0, 0]
    return (n, c, h, w, cp["weight"][0], cp["weight"][2], cp["stride"][0], pad[0], cp["dilation"][0], cp["groups"])


def ws_fn(cp, ob):
    k = pkey(cp)
    if k not in PROBE:
        MISSING.add(k)
        return 0
    return PROBE[k]


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
        for name, f in (("step1", None), ("probe_table_3090", ws_fn)):
            p = replay4.predict(lab, {"eff": prof["capacity_bytes"]}, drop="conv", ws_fn=f, ws_rel=pos.get(c), ws_bytes=wsb)
            a = p["R_at"]["eff"]
            out[f"{g}|{name}"] = None if a["oom"] else a["reserved"]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(8) as pool:
        rows = pool.map(run, cases)
    (OUT / "probe_table_replays.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    rep = {"probe_signatures": len(PROBE), "result": {}}
    for g in gpus:
        for name in ("step1", "probe_table_3090"):
            errs = []
            for r in rows:
                m = meas.get((g, r["case_id"]))
                v = r[f"{g}|{name}"]
                if m and not m[1] and v is not None:
                    errs.append(v - m[0])
            rep["result"][f"{g}|{name}"] = {
                "n": len(errs), "within100MB": sum(abs(e) <= 1e8 for e in errs),
                "exact": sum(e == 0 for e in errs), "median_abs_MiB": round(statistics.median(abs(e) for e in errs) / MIB, 1),
                "max_over_MiB": round(max(errs) / MIB, 1), "max_under_MiB": round(min(errs) / MIB, 1)}
    (OUT / "probe_table_summary.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
