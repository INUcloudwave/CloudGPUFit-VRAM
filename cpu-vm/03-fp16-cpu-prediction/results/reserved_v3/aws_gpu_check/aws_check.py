#!/usr/bin/env python3
"""Indirect check of the descending segment-address rule on AWS GPUs (stdlib only, read-only on inputs).

Idea: the rule was confirmed only on RTX 3090 GPU traces. AWS runs have no allocator traces, only per-condition
measured values. For conditions where the GPU request stream is the same as on the RTX 3090 (same measured start and
peak *allocated*), the reserved peak is decided by the allocator + segment addresses alone. Then:
  * AWS reserved == RTX 3090 reserved  -> the GPU's allocator/address behaviour gives the same result as the 3090
  * in particular on "discriminating" conditions, where stock xMem addresses and descending addresses give different
    predictions (prediction file ablate.stock_nows vs ablate.desc_nows), and the 3090 measurement matched descending.
Also: AWS measured reserved vs v3 prediction (descending + cuBLAS workspace, R_inf) on no-retry conditions.

usage: aws_check.py <aws extract root> <v3 predictions.jsonl> <3090 L256 csv> <out dir>
"""
import csv
import glob
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

MIB = 1 << 20
TOL = 100_000_000
CNN = {"resnet-18", "resnet-50", "mobilenet-v2-1.0-224"}


def load_aws(root):
    gpus = {}
    for d in sorted(os.listdir(root)):
        fs = glob.glob(f"{root}/{d}/runs/*/*/measure.json")
        if not fs:
            continue
        rows, envs = {}, Counter()
        for f in fs:
            m = json.load(open(f))
            cid = os.path.basename(os.path.dirname(f))
            e = m["environment"]
            envs[(e["gpu_name"], tuple(e["capability"]), e["cuda_total_memory"], e["driver"], e["torch"], e["cudnn"],
                  e["image_id"][:19])] += 1
            r = {"status": m["status"], "model": cid.split("__")[0]}
            if m["status"] == "ok":
                m0 = m["measures"][0]
                s = m["summary"]
                r.update(res=s["first_measure_peak_reserved"], alloc=s["first_measure_peak_allocated"],
                         start_alloc=s["first_measure_start_allocated"], start_res=s["first_measure_start_reserved"],
                         load_alloc=m["after_load"]["allocated_bytes.all.current"],
                         retries=m0["stats_end"]["num_alloc_retries"],
                         outside=m0["nvml_process_peak_bytes"] - m0["peak"]["reserved"]
                         if m0.get("nvml_process_peak_bytes") else None)
            rows[cid] = r
        gpus[d] = {"rows": rows, "env": envs}
    return gpus


def main():
    root, pred_p, csv_p, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    pred = {}
    for line in open(pred_p):
        p = json.loads(line)
        if p.get("status") == "ok" and p["table"] in ("both", "L256"):
            pred[p["case_id"]] = p
    g3090 = {}
    for r in csv.DictReader(open(csv_p)):
        if r["measure_status"] == "ok":
            g3090[r["condition_id"]] = {"res": int(r["m1_peak_reserved"]), "alloc": int(r["m1_peak_allocated"]),
                                        "start_alloc": int(r["m1_start_allocated"]),
                                        "load_alloc": int(r["after_load_allocated"])}
    gpus = load_aws(root)
    summary, per = {}, []
    for g, G in gpus.items():
        rows = G["rows"]
        st = Counter(r["status"] for r in rows.values())
        ok = {c: r for c, r in rows.items() if r["status"] == "ok"}
        outs = [r["outside"] for r in ok.values() if r["outside"] is not None]
        env = G["env"].most_common()
        cuda_total = env[0][0][2]
        o_med = statistics.median(outs)
        # cuBLAS workspace: start_alloc - load_alloc on decoder conditions vs RTX 3090 (same stream -> same tensors)
        wsd = Counter()
        for c, r in ok.items():
            t = g3090.get(c)
            if t and pred.get(c, {}).get("kind") == "decoder_llm":
                wsd[(r["start_alloc"] - r["load_alloc"]) - (t["start_alloc"] - t["load_alloc"])] += 1
        cls = Counter()
        disc = Counter()
        acc = Counter()
        for c, r in ok.items():
            t, p = g3090.get(c), pred.get(c)
            if not t or not p:
                cls[("no_3090_or_pred",)] += 1
                continue
            same_stream = r["alloc"] == t["alloc"] and r["start_alloc"] == t["start_alloc"]
            retry = r["retries"] > 0
            cnn = r["model"] in CNN
            res_eq = r["res"] == t["res"]
            key = ("retry" if retry else "noretry", "cnn" if cnn else "other",
                   "same_alloc" if same_stream else "diff_alloc", "res_eq" if res_eq else "res_diff")
            cls[key] += 1
            ab = p.get("ablate", {})
            stock, desc = ab.get("stock_nows", {}).get("R_inf"), ab.get("desc_nows", {}).get("R_inf")
            rinf = p["main"]["R_inf"]
            is_disc = stock is not None and desc is not None and stock != desc
            if is_disc and same_stream and not retry:
                disc[("3090_eq_desc+ws" if t["res"] == rinf else "3090_ne_desc+ws",
                      "aws_eq_3090" if res_eq else "aws_ne_3090",
                      "aws_eq_desc+ws" if r["res"] == rinf else "aws_ne_desc+ws")] += 1
            if not retry:
                grp = "cnn" if cnn else "other"
                acc[(grp, "n")] += 1
                acc[(grp, "within100MB")] += abs(rinf - r["res"]) <= TOL
                acc[(grp, "exact")] += rinf == r["res"]
                acc[(grp, "3090_within100MB")] += abs(rinf - t["res"]) <= TOL
            per.append({"gpu": g, "case_id": c, "model": r["model"], "retry": retry, "same_alloc_as_3090": same_stream,
                        "aws_res": r["res"], "rtx3090_res": t["res"], "v3_R_inf": rinf, "stock_nows": stock,
                        "desc_nows": desc, "aws_minus_3090_MiB": round((r["res"] - t["res"]) / MIB, 1),
                        "v3_minus_aws_MiB": round((rinf - r["res"]) / MIB, 1)})
        summary[g] = {
            "status": dict(st), "env": [[list(map(str, k)), n] for k, n in env],
            "cuda_total_memory": cuda_total, "outside_median": o_med, "outside_min": min(outs), "outside_max": max(outs),
            "capacity_bytes": cuda_total - o_med,
            "ws_diff_vs_3090_bytes": {str(k): v for k, v in wsd.most_common(5)},
            "classes": {"|".join(k): v for k, v in sorted(cls.items())},
            "discriminating": {"|".join(k): v for k, v in sorted(disc.items())},
            "accuracy_noretry": {f"{a}|{b}": v for (a, b), v in sorted(acc.items())},
        }
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
    with open(f"{out}/per_condition.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0]))
        w.writeheader()
        w.writerows(per)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
