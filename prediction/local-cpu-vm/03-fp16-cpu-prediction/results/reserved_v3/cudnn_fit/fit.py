"""Step 1 + 2 (2026-11-20): CPU library-layer removal and a per-GPU convolution-workspace coefficient, vision models.

Step 1 (CPU only): R_inf/R_at of v3 vs drop="conv" vs drop="all" (cgf680.replay4) against every measured GPU.
Step 2 (per-GPU coefficient from existing measurements, no GPU trace): conv workspace = r x feature for every conv
call, feature = output bytes ("out") or input bytes ("in"); r from a grid, chosen per GPU to minimise the median
absolute error; evaluated in-sample and leave-one-model-out (fit r on the other models, test on the held-out one).
Run inside the pinned image:
  MEM=16g CPUS=8 docs/run_in_image.sh results/reserved_v3/cudnn_fit/fit.py <per_condition.csv> <out dir>
"""
import csv
import json
import statistics
import sys
from multiprocessing import Pool
from pathlib import Path

from cgf680 import replay4
from cgf680.gpu_rules import cublas_workspace

E = Path("/exp")
PERCOND, OUT = sys.argv[1], Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
MIB = 1 << 20
TOL = 100_000_000
CNN = ("resnet-18", "resnet-50", "mobilenet-v2-1.0-224")
VISION = CNN + ("vit-large-patch16-224",)
GPU_PROFILE = {"rtx3090": "rtx3090", "g4dn.xlarge-l256": "t4", "g5.xlarge-l256": "a10g", "g6.xlarge-l256": "l4",
               "gr6.4xlarge-l256": "l4", "g6e.xlarge-l256": "l40s", "g7.2xlarge__l256__cu128": "rtxpro4500"}
GRID = [x / 4 for x in range(0, 33)]          # r = 0 .. 8 step 0.25

profiles = json.loads((E / "manifest/gpu_profiles.json").read_text())["profiles"]
pos = {}
for line in (E / "runs/gemm_exact/positions.jsonl").read_text().splitlines():
    r = json.loads(line)
    if r.get("status") == "ok" and r.get("prefix_equal_to_npz"):
        pos[r["case_id"]] = r["ws_insert_rel"]

meas = {}                                       # (gpu, case) -> (reserved, retry)
r3090 = {}
for r in csv.DictReader(open(PERCOND)):
    if r["model"] not in VISION:
        continue
    meas[(r["gpu"], r["case_id"])] = (int(r["aws_res"]), r["retry"] == "True")
    r3090[r["case_id"]] = int(r["rtx3090_res"])
for c, v in r3090.items():
    meas[("rtx3090", c)] = (v, False)            # 3090 retry conditions are LLM/T5 only (alloc_retry_conditions.csv)
cases = sorted({c for _, c in meas if (E / f"runs/layer_labels/{c}.npz").exists()})
gpus = sorted({g for g, _ in meas})


def feat(kind):
    if kind == "out":
        return lambda cp, ob: ob
    return lambda cp, ob: 2 * _prod(cp["input"])     # FP16 input bytes


def _prod(s):
    p = 1
    for x in s:
        p *= x
    return p


def run_case(c):
    lab = replay4.load_labels(E / f"runs/layer_labels/{c}.npz")
    res = {"case_id": c, "model": c.split("__")[0], "variants": {}, "grid": {}}
    for g in gpus:
        prof = profiles[GPU_PROFILE[g]]
        wsb = cublas_workspace(prof["gpu_name"], prof.get("compute_capability"), prof.get("runtime_torch", "2.6.0"))["bytes"]
        caps = {"eff": prof["capacity_bytes"]}
        for drop in ("none", "conv", "all"):
            p = replay4.predict(lab, caps, drop=drop, ws_rel=pos.get(c), ws_bytes=wsb)
            res["variants"][f"{g}|{drop}"] = {"R_inf": p["R_inf"], "R_at": p["R_at"]["eff"], "dropped_bytes": p["dropped_bytes"]}
        if c.split("__")[0] in CNN:
            for kind in ("out", "in"):
                f = feat(kind)
                for r in GRID:
                    p = replay4.predict(lab, caps, drop="conv", ws_fn=(lambda cp, ob, r=r, f=f: r * f(cp, ob)),
                                        ws_rel=pos.get(c), ws_bytes=wsb)
                    a = p["R_at"]["eff"]
                    res["grid"][f"{g}|{kind}|{r}"] = None if a["oom"] else a["reserved"]
    return res


def main():
    with Pool(8) as pool:
        rows = pool.map(run_case, cases)
    (OUT / "replays.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    rep = {"step1": {}, "step2": {}}
    # step 1
    for g in gpus:
        for grp in ("cnn", "vit"):
            for drop in ("none", "conv", "all"):
                errs = []
                for r in rows:
                    if (grp == "cnn") != (r["model"] in CNN):
                        continue
                    m = meas.get((g, r["case_id"]))
                    if not m or m[1]:
                        continue
                    a = r["variants"][f"{g}|{drop}"]["R_at"]
                    if a["oom"]:
                        continue
                    errs.append(a["reserved"] - m[0])
                if errs:
                    rep["step1"][f"{g}|{grp}|{drop}"] = {
                        "n": len(errs), "within100MB": sum(abs(e) <= TOL for e in errs),
                        "median_MiB": round(statistics.median(errs) / MIB, 1),
                        "max_over_MiB": round(max(errs) / MIB, 1), "max_under_MiB": round(min(errs) / MIB, 1)}
    # step 2
    cnn_rows = [r for r in rows if r["model"] in CNN]
    for g in gpus:
        for kind in ("out", "in"):
            def errs_for(rr, rs):
                out = []
                for r in rs:
                    m = meas.get((g, r["case_id"]))
                    v = r["grid"].get(f"{g}|{kind}|{rr}")
                    if m and not m[1] and v is not None:
                        out.append(v - m[0])
                return out

            def score(rr, rs):
                e = errs_for(rr, rs)
                return statistics.median([abs(x) for x in e]) if e else float("inf")
            best = min(GRID, key=lambda rr: score(rr, cnn_rows))
            e_in = errs_for(best, cnn_rows)
            loo = []
            for held in CNN:
                train = [r for r in cnn_rows if r["model"] != held]
                test = [r for r in cnn_rows if r["model"] == held]
                rb = min(GRID, key=lambda rr: score(rr, train))
                loo += [(held, rb, x) for x in errs_for(rb, test)]
            base = errs_for(0.0, cnn_rows)
            rep["step2"][f"{g}|{kind}"] = {
                "best_r": best,
                "r0_within100MB": sum(abs(x) <= TOL for x in base), "n": len(base),
                "in_sample_within100MB": sum(abs(x) <= TOL for x in e_in),
                "in_sample_median_abs_MiB": round(statistics.median([abs(x) for x in e_in]) / MIB, 1),
                "loo_within100MB": sum(abs(x[2]) <= TOL for x in loo),
                "loo_r_by_heldout": sorted({(h, rb) for h, rb, _ in loo}),
                "loo_median_abs_MiB": round(statistics.median([abs(x[2]) for x in loo]) / MIB, 1),
                "loo_max_under_MiB": round(min(x[2] for x in loo) / MIB, 1),
                "loo_max_over_MiB": round(max(x[2] for x in loo) / MIB, 1)}
    (OUT / "summary.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
