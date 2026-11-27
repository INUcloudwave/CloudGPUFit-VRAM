#!/usr/bin/env python3
"""Compare reserved prediction v3 (tools/reserved_predict.py) with RTX 3090 measurements (host side, stdlib only).

GPU reference: m1_peak_reserved of measure_status=ok rows (L256 and L1024 per-condition CSVs; L256 preferred when a
condition is in both, the other value kept for the GPU self-variance check). Retry conditions (num_alloc_retries > 0)
from INUGPU alloc_retry_conditions.csv are reported separately, CNNs (cuDNN conv path, not modelled) too.
Tolerance: |pred - GPU| <= 100 MB (1e8 B, agreed project tolerance; --tol-bytes); 32 MiB (INUGPU guide) also reported.
  python3 tools/compare_reserved.py --pred results/reserved_v3/rtx3090_v3/predictions.jsonl \
     --gpu-l256 <L256 csv> --gpu-l1024 <L1024 csv> --retry <alloc_retry_conditions.csv> --out-prefix handoff_codex/reserved_prediction
"""
import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

MIB = 1 << 20
TOL = 100_000_000          # 100 MB, project tolerance (overridable with --tol-bytes)
TOL32 = 32 * MIB           # INUGPU guide tolerance, reported alongside
CNN = {"resnet-18", "resnet-50", "mobilenet-v2-1.0-224"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


GPU_ALLOC = {}


def read_gpu(path):
    out, oom = {}, {}
    for r in csv.DictReader(open(path)):
        if r["measure_status"] == "ok":
            out[r["condition_id"]] = int(r["m1_peak_reserved"])
            GPU_ALLOC.setdefault(r["condition_id"], int(r["m1_peak_allocated"]))
        else:
            oom[r["condition_id"]] = r["measure_status"]
    return out, oom


def stats(errs):
    if not errs:
        return {"n": 0}
    a = sorted(abs(e) for e in errs)
    return {"n": len(errs), "within_tol": sum(x <= TOL for x in a), "share_within_tol": round(sum(x <= TOL for x in a) / len(a), 4),
            "within_32MiB": sum(x <= TOL32 for x in a), "share_within_32MiB": round(sum(x <= TOL32 for x in a) / len(a), 4),
            "exact_0": sum(x == 0 for x in a), "median_abs_MiB": round(statistics.median(a) / MIB, 1),
            "p90_abs_MiB": round(a[min(len(a) - 1, int(0.9 * len(a)))] / MIB, 1), "max_abs_MiB": round(a[-1] / MIB, 1),
            "mean_signed_MiB": round(statistics.mean(errs) / MIB, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gpu-l256", required=True)
    ap.add_argument("--gpu-l1024", required=True)
    ap.add_argument("--retry", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--tol-bytes", type=int, default=TOL)
    a = ap.parse_args()
    globals()["TOL"] = a.tol_bytes
    g256, oom256 = read_gpu(a.gpu_l256)
    g1024, oom1024 = read_gpu(a.gpu_l1024)
    retry = defaultdict(dict)
    for r in csv.DictReader(open(a.retry)):
        retry[r["condition_id"]][r["run"]] = int(r["num_alloc_retries"])
    preds = {}
    for line in open(a.pred):
        r = json.loads(line)
        if r.get("status") == "ok":
            preds[r["case_id"]] = r
    rows = []
    for cid, p in sorted(preds.items()):
        gpu = g256.get(cid, g1024.get(cid))
        other = g1024.get(cid) if cid in g256 else None
        m = p["main"]
        eff = m["R_at"]["effective"]
        row = {"condition_id": cid, "model": p["model"], "table": p["table"], "ws_rel": p["ws_rel"],
               "gpu_m1_peak_reserved": gpu, "gpu_other_table_value": other, "gpu_m1_peak_allocated": GPU_ALLOC.get(cid),
               "gpu_status": "ok" if gpu is not None else (oom256.get(cid) or oom1024.get(cid) or "not_measured"),
               "gpu_retry": bool(retry.get(cid)), "gpu_retry_counts": json.dumps(retry.get(cid)) if cid in retry else "",
               "pred_R_at_effective": eff.get("reserved"), "pred_oom_at_effective": eff.get("oom"),
               "pred_retry_at_effective": eff.get("retry"), "pred_R_at_cuda_total": m["R_at"]["cuda_total"].get("reserved"),
               "pred_R_inf": m["R_inf"], "pred_R_min": m.get("R_min"), "pred_max_allocated": m["max_allocated"],
               "v2_first_measure_peak_reserved": p["v2"]["first_measure_peak_reserved"],
               "abl_stock_nows_R_at_effective": p.get("ablate", {}).get("stock_nows", {}).get("R_at_effective", {}).get("reserved"),
               "abl_desc_nows_R_at_effective": p.get("ablate", {}).get("desc_nows", {}).get("R_at_effective", {}).get("reserved")}
        if gpu is not None:
            for k in ("pred_R_at_effective", "v2_first_measure_peak_reserved", "abl_stock_nows_R_at_effective",
                      "abl_desc_nows_R_at_effective", "pred_R_at_cuda_total"):
                v = row[k]
                row["err_" + k] = (v - gpu) if v is not None else None
            e = row["err_pred_R_at_effective"]
            if row["gpu_retry"]:
                cls = "retry"
            elif p["model"] in CNN:
                cls = "cnn"
            else:
                cls = "normal"
            row["group"] = cls
            if e is None:
                cause = "pred_oom_but_gpu_ok"
            elif abs(e) <= TOL:
                cause = "within_tol"
            elif p["model"] not in CNN and abs(GPU_ALLOC.get(cid, -10**15) - m["max_allocated"]) <= 4 * MIB:
                # same peak allocated bytes as the GPU (within 4 MiB: cuBLASLt transients): the request amounts are right, the gap is block reuse / placement
                cause = ("retry (capacity-bound, allocated matches)" if cls == "retry"
                         else "reuse_or_placement (allocated matches GPU)")
            elif cls == "retry":
                cause = "retry (capacity-bound)"
            elif cls == "cnn":
                cause = "cnn_conv_path (cuDNN vs mkldnn requests)"
            elif other is not None and other != gpu and min(gpu, other) - TOL <= gpu + e <= max(gpu, other) + TOL:
                cause = "gpu_self_variance"
            else:
                cause = "unresolved (request list / address)"
            row["cause"] = cause
        else:
            row["group"] = "gpu_not_ok"
            row["cause"] = ("pred_oom_matches_gpu_oom" if eff.get("oom") else "pred_runs_but_gpu_oom") \
                if row["gpu_status"].startswith("gpu_oom") else "gpu_not_ok"
        rows.append(row)
    # summary
    S = {"tolerance_bytes": TOL, "tolerance_32MiB_bytes": TOL32, "inputs": {"pred": a.pred, "gpu_l256": [a.gpu_l256, sha(a.gpu_l256)], "gpu_l1024": [a.gpu_l1024, sha(a.gpu_l1024)],
                    "retry": [a.retry, sha(a.retry)]},
         "gpu_ok_conditions_union": len(set(g256) | set(g1024)), "predicted_conditions": len(preds),
         "compared": sum(r["gpu_m1_peak_reserved"] is not None for r in rows),
         "gpu_ok_without_cpu_prediction": sorted((set(g256) | set(g1024)) - set(preds))}
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["gpu_m1_peak_reserved"] is None:
            continue
        for k in ("pred_R_at_effective", "v2_first_measure_peak_reserved", "abl_stock_nows_R_at_effective",
                  "abl_desc_nows_R_at_effective", "pred_R_at_cuda_total"):
            e = r.get("err_" + k)
            if e is not None:
                by[r["group"]][k].append(e)
                by["all"][k].append(e)
    S["by_group"] = {g: {k: stats(v) for k, v in d.items()} for g, d in by.items()}
    bym = defaultdict(list)
    for r in rows:
        if r["group"] in ("normal", "cnn") and r.get("err_pred_R_at_effective") is not None:
            bym[r["model"]].append(r["err_pred_R_at_effective"])
    S["normal_cnn_by_model_v3"] = {m: stats(v) for m, v in sorted(bym.items())}
    S["causes"] = defaultdict(int)
    for r in rows:
        S["causes"][r["cause"]] += 1
    # retry detection: predicted retry (OOM-retry needed at effective capacity) vs GPU retry list
    cm = defaultdict(int)
    for r in rows:
        if r["gpu_m1_peak_reserved"] is None:
            continue
        cm[f"gpu_retry={r['gpu_retry']} pred_retry={bool(r['pred_retry_at_effective'])}"] += 1
    S["retry_confusion"] = dict(cm)
    S["gpu_oom_with_cpu_prediction"] = [{k: r[k] for k in ("condition_id", "gpu_status", "pred_oom_at_effective",
                                                         "pred_R_min", "pred_R_inf")} for r in rows if r["group"] == "gpu_not_ok"]
    S["worst_normal_v3"] = sorted(({"condition_id": r["condition_id"], "err_MiB": r["err_pred_R_at_effective"] / MIB,
                                    "gpu_MiB": r["gpu_m1_peak_reserved"] / MIB, "cause": r["cause"]}
                                   for r in rows if r["group"] == "normal" and r.get("err_pred_R_at_effective") is not None),
                                  key=lambda x: -abs(x["err_MiB"]))[:15]
    out = Path(a.out_prefix)
    cols = list(rows[0].keys()) + [k for k in ("err_pred_R_at_effective", "err_v2_first_measure_peak_reserved",
                                               "err_abl_stock_nows_R_at_effective", "err_abl_desc_nows_R_at_effective",
                                               "err_pred_R_at_cuda_total", "group", "cause") if k not in rows[0]]
    seen = []
    for c in cols:
        if c not in seen:
            seen.append(c)
    with open(f"{out}_per_condition.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=seen, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    Path(f"{out}_summary.json").write_text(json.dumps(S, indent=1, default=dict))
    print(json.dumps({k: S[k] for k in ("compared", "predicted_conditions", "gpu_ok_conditions_union", "causes",
                                        "retry_confusion")}, indent=1, default=dict))
    for g in ("normal", "cnn", "retry", "all"):
        if g in S["by_group"]:
            print(g, {k: v for k, v in S["by_group"][g].items()})


if __name__ == "__main__":
    main()
