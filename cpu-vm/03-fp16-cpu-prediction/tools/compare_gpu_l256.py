#!/usr/bin/env python3
"""CPU predictions (results/<run>/cpu_predictions.jsonl from collect395) + optional capacity replay
(results/<run>/capacity_<gpu>.jsonl) vs a GPU per-condition CSV in the INUGPU export_for_inuos format (bytes).

  python3 tools/compare_gpu_l256.py --run main395_l256 \
     --gpu-csv /home/inuos/cgf-fp16-inuos-handoff-20261115/results/rtx3090_main395_L256/<per_condition>.csv \
     --gpu-label rtx3090_L256 --capacity rtx3090
Writes results/<run>/compare_<label>.csv and compare_<label>_summary.md. All values bytes; MiB only in the md.
"""
import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

EXP = Path("/home/inuos/cgf-fp16-680-20261111")
MiB = 2 ** 20
KEYS = [("first_measure_peak_reserved", "m1_peak_reserved"), ("first_measure_peak_allocated", "m1_peak_allocated"),
        ("first_measure_start_reserved", "m1_start_reserved"), ("after_load_reserved", "after_load_reserved"),
        ("after_load_allocated", "after_load_allocated")]


def num(v):
    return int(float(v)) if v not in (None, "") else None


def rel(c, g):
    return (c - g) / g if c is not None and g else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main395_l256")
    ap.add_argument("--gpu-csv", required=True)
    ap.add_argument("--gpu-label", required=True)
    ap.add_argument("--capacity", default=None, help="gpu profile name of results/<run>/capacity_<name>.jsonl")
    a = ap.parse_args()
    res = EXP / "results" / a.run
    cpu = {r["condition_id"]: r for r in map(json.loads, (res / "cpu_predictions.jsonl").read_text().splitlines())}
    gpu = {r["condition_id"]: r for r in csv.DictReader(open(a.gpu_csv))}
    cap = {}
    if a.capacity:
        for l in (res / f"capacity_{a.capacity}.jsonl").read_text().splitlines():
            x = json.loads(l)
            if "error" not in x:
                cap[x["condition_id"]] = x
    rows = []
    for cid in sorted(set(cpu) | set(gpu)):
        c, g, k = cpu.get(cid, {}), gpu.get(cid, {}), cap.get(cid)
        x = {"condition_id": cid, "model": g.get("model") or c.get("model"), "workload_type": g.get("workload_type") or c.get("workload_type"),
             "cpu_status": c.get("status"), "gpu_status": g.get("measure_status"), "gpu_failed_stage": g.get("measure_failed_stage"),
             "gpu_attn": g.get("attn_implementation"), "cpu_attn": c.get("attn_implementation")}
        gs = dict(kv.split("=", 1) for kv in g["input_sha256"].split(";")) if g.get("input_sha256") else None
        cs = {kk[6:-7]: c[kk] for kk in c if kk.startswith("input_") and kk.endswith("_sha256")} if c else None
        x["input_sha_same"] = (gs == cs) if gs and cs else None
        for ck, gk in KEYS:
            cv, gv = c.get(ck), num(g.get(gk))
            x[f"cpu1TiB_{ck}"], x[f"gpu_{ck}"], x[f"rel1TiB_{ck}"] = cv, gv, rel(cv, gv)
        if k:
            x["cap_oom_predicted"], x["cap_oom_stage"] = k["oom_predicted"], k["oom_stage"]
            x["cap_cache_release_events"] = k["cache_release_events"]
            for ck in ("first_measure_peak_reserved", "first_measure_peak_allocated", "first_measure_start_reserved"):
                x[f"cap_{ck}"] = k.get(ck)
                x[f"relcap_{ck}"] = rel(k.get(ck), num(g.get(dict(KEYS)[ck])))
        gpu_oom = g.get("measure_status") == "gpu_oom"
        if c.get("status") == "ok" and k is not None:
            x["oom_agreement"] = "both_oom" if (k["oom_predicted"] and gpu_oom) else "both_fit" if (not k["oom_predicted"] and g.get("measure_status") == "ok") \
                else "cpu_oom_only" if k["oom_predicted"] else "gpu_oom_only" if gpu_oom else None
        elif c.get("status") in ("worker_killed_cpu_oom", "worker_killed_cpu_oom_suspected", "cpu_oom", "preempted_low_mem"):
            x["oom_agreement"] = "cpu_not_predictable_ram"
        x["comparable"] = c.get("status") == "ok" and g.get("measure_status") == "ok"
        rows.append(x)
    cols = []
    for r in rows:
        cols += [kk for kk in r if kk not in cols]
    out = res / f"compare_{a.gpu_label}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    # summary
    L = [f"# CPU vs {a.gpu_label} ({a.run})", "",
         f"- conditions: CPU {len(cpu)}, GPU {len(gpu)}, both ok {sum(r['comparable'] for r in rows)}",
         f"- CPU status: {dict(Counter(r['cpu_status'] for r in rows))}",
         f"- GPU status: {dict(Counter(r['gpu_status'] for r in rows))}",
         f"- input sha256 same: {sum(1 for r in rows if r['input_sha_same'])} / {sum(1 for r in rows if r['input_sha_same'] is not None)}",
         f"- OOM agreement (capacity replay): {dict(Counter(r.get('oom_agreement') for r in rows if r.get('oom_agreement')))}", ""]
    for label, pref in (("1 TiB replay (worker value)", "rel1TiB_"), ("capacity replay", "relcap_")):
        for ck in ("first_measure_peak_reserved", "first_measure_peak_allocated"):
            by = defaultdict(list)
            for r in rows:
                v = r.get(pref + ck)
                if r["comparable"] and v is not None:
                    by[r["model"].split("/")[-1]].append(v)
            if not by:
                continue
            L += [f"## {ck} — {label}", "", "| model | n | ≤1% | ≤5% | ≤20% | >20% | median | min | max |", "|---|---|---|---|---|---|---|---|---|"]
            for m, v in by.items():
                aa = [abs(t) for t in v]
                L.append(f"| {m} | {len(v)} | {sum(t <= .01 for t in aa)} | {sum(t <= .05 for t in aa)} | {sum(t <= .2 for t in aa)} | "
                         f"{sum(t > .2 for t in aa)} | {statistics.median(v):+.3f} | {min(v):+.3f} | {max(v):+.3f} |")
            L.append("")
    (res / f"compare_{a.gpu_label}_summary.md").write_text("\n".join(L))
    print("\n".join(L[:8]))


if __name__ == "__main__":
    main()
