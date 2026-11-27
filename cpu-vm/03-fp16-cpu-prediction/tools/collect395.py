#!/usr/bin/env python3
"""Collect runs/<run>/cases/*/result.json (worker v2) -> results/<run>/:
  cpu_predictions.csv / .jsonl   one row per condition_id, GPU-aligned keys (bytes, int)
  status_summary.json            counts by status / phase, failure list (stage), retry history
  pilot_vs_rtx3090.csv           pilot rows joined with the RTX 3090 default-attention pilot reference
  compare_<label>.csv            with --gpu <summary.csv|jsonl> (GPU main results, condition_id key)
"""
import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

EXP = Path("/home/inuos/cgf-fp16-680-20261111")
REF = Path("/home/inuos/cgf-fp16-inuos-handoff-20261112/reference/pilot_rtx3090_eager_vs_default.csv")
MiB = 2 ** 20
SUM_KEYS = ["first_measure_start_reserved", "first_measure_start_allocated", "first_measure_peak_reserved",
            "first_measure_peak_allocated", "first_measure_end_with_output_reserved", "first_measure_end_with_output_allocated",
            "first_measure_released_reserved", "first_measure_released_allocated", "after_load_reserved",
            "after_load_allocated", "load_peak_allocated"]


def row(r, cdir):
    c = r.get("case", {})
    s = r.get("summary") or {}
    v = r.get("variant_with_cublas_workspace") or {}
    rn = r.get("runner") or {}
    t = r.get("timings_s") or {}
    mi = r.get("model_info") or {}
    out = {"condition_id": r.get("condition_id"), "model": r.get("model"), "revision": r.get("revision"),
           "workload_type": c.get("kind"), "batch_size": c.get("batch_size"), "input_len": c.get("input_len"),
           "max_new_tokens": c.get("new_tokens"), "resolution": c.get("resolution"), "pilot": c.get("pilot"),
           "status": r.get("status"), "failed_stage": r.get("failed_stage"), "error": (r.get("error") or "")[:300],
           "phase": rn.get("phase"), "attempt": rn.get("attempt"), "new_tokens": r.get("new_tokens"),
           "attn_implementation": mi.get("attn_implementation"), "dtype_mode": r.get("dtype_mode")}
    for k, x in (r.get("inputs") or {}).items():
        out[f"input_{k}_sha256"] = x["sha256"]
        out[f"input_{k}_shape"] = "x".join(map(str, x["shape"]))
    for k in SUM_KEYS:
        out[k] = s.get(k)
    for k in ("first_measure_peak_reserved", "first_measure_peak_allocated", "first_measure_start_reserved",
              "first_measure_start_allocated"):
        out[f"ws_variant_{k}"] = v.get(k)
    fe = r.get("fp16_emulation") or {}
    out["fp32_request_bytes_per_invocation"] = fe.get("fp32_request_bytes")
    out["fp16_request_bytes_per_invocation"] = fe.get("fp16_request_bytes")
    out["emulated_upcasts"] = (fe.get("emulated_casts") or {}).get("upcast")
    out["emulated_downcasts"] = (fe.get("emulated_casts") or {}).get("downcast")
    out["attention_ops_cpu"] = json.dumps(r.get("attention_ops_cpu"))[:500] if r.get("attention_ops_cpu") else None
    for k in ("model_load", "trace_invocation", "trace_export", "trace_parse", "shadow_match", "xmem_replay", "attn_probe", "total_worker"):
        out[f"time_{k}_s"] = round(t[k], 2) if isinstance(t.get(k), (int, float)) else None
    out["time_container_wall_s"] = round(rn["wall_seconds"], 2) if rn.get("wall_seconds") else None
    out["worker_max_rss_bytes"] = (r.get("rss") or {}).get("max_rss_bytes")
    out["container_memory_peak_bytes"] = rn.get("container_memory_peak_bytes")
    out["host_mem_available_before_bytes"] = (rn.get("host_mem_before") or {}).get("MemAvailable")
    out["cpuset"], out["threads"], out["mem_limit"] = rn.get("cpuset"), rn.get("threads"), rn.get("mem_limit")
    out["image_id"] = rn.get("image_id")
    env = r.get("environment") or {}
    out["torch"], out["transformers"] = env.get("torch"), env.get("transformers")
    out["xmem_commit"] = (r.get("xmem") or {}).get("commit")
    out["result_json"] = str(cdir / "result.json")
    out["requests_npz"] = str(cdir / "invocation_fp16_requests.npz") if (cdir / "invocation_fp16_requests.npz").exists() else None
    out["container_log"] = str(cdir / "container.log")
    return out


def write_csv(path, rows):
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def rel(a, b):
    return None if a is None or not b else (a - b) / b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="main395")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--gpu-label", default=None)
    a = ap.parse_args()
    root = EXP / "runs" / a.run_name
    out = EXP / "results" / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cdir in sorted(p for p in (root / "cases").iterdir() if p.is_dir() and ".attempt-" not in p.name):
        if (cdir / "result.json").exists():
            rows.append(row(json.loads((cdir / "result.json").read_text()), cdir))
    write_csv(out / "cpu_predictions.csv", rows)
    (out / "cpu_predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    attempts = [json.loads(l) for l in (root / "attempts.jsonl").read_text().splitlines()] if (root / "attempts.jsonl").exists() else []
    summ = {"run": a.run_name, "conditions_with_result": len(rows), "status": dict(Counter(r["status"] for r in rows)),
            "status_by_phase": dict(Counter(f"{r['phase']}:{r['status']}" for r in rows)),
            "failures": [{k: r[k] for k in ("condition_id", "status", "failed_stage", "phase", "error")} for r in rows if r["status"] != "ok"],
            "attempt_history": [x for x in attempts if x.get("status") != "ok"]}
    (out / "status_summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    # pilot vs RTX 3090 reference (default attention)
    ref = {r["condition_id"]: r for r in csv.DictReader(REF.open())}
    prow = []
    for r in rows:
        g = ref.get(r["condition_id"])
        if not g or r["status"] != "ok":
            continue
        x = {"condition_id": r["condition_id"], "gpu_attn": g["default_attn_impl"], "cpu_attn": r["attn_implementation"],
             "input_sha_same": json.loads(g["default_input_sha256"]) == {k[6:-7]: r[k][:16] for k in r if k.startswith("input_") and k.endswith("_sha256")}}
        for key, gk in (("after_load_reserved", "default_load_reserved_MiB"), ("load_peak_allocated", "default_load_peak_allocated_MiB"),
                        ("first_measure_start_reserved", "default_m1_start_reserved_MiB"),
                        ("first_measure_start_allocated", "default_m1_start_allocated_MiB"),
                        ("first_measure_peak_reserved", "default_m1_peak_reserved_MiB"),
                        ("first_measure_peak_allocated", "default_m1_peak_allocated_MiB")):
            cpu, gpu = r[key] / MiB, float(g[gk])
            x[f"{key}_cpu_MiB"], x[f"{key}_gpu_MiB"] = round(cpu, 1), gpu
            x[f"{key}_diff_MiB"], x[f"{key}_rel"] = round(cpu - gpu, 1), round(rel(cpu, gpu), 4) if gpu else None
        for key, gk in (("first_measure_peak_reserved", "default_m1_peak_reserved_MiB"), ("first_measure_peak_allocated", "default_m1_peak_allocated_MiB")):
            v = r[f"ws_variant_{key}"]
            if v is not None:
                x[f"ws_variant_{key}_rel"] = round(rel(v / MiB, float(g[gk])), 4)
        prow.append(x)
    if prow:
        write_csv(out / "pilot_vs_rtx3090.csv", prow)
    print(json.dumps({k: summ[k] for k in ("run", "conditions_with_result", "status", "status_by_phase")}, ensure_ascii=False))
    if prow:
        for key in ("first_measure_peak_reserved", "first_measure_peak_allocated"):
            e = [abs(p[f"{key}_rel"]) for p in prow]
            print(key, "pilot n", len(e), "mean |rel|", round(statistics.mean(e) * 100, 2), "% max", round(max(e) * 100, 1), "%")
    if a.gpu:  # GPU main results: condition_id + summary keys (bytes) in csv/jsonl
        gp = Path(a.gpu)
        grows = [json.loads(l) for l in gp.read_text().splitlines()] if gp.suffix == ".jsonl" else list(csv.DictReader(gp.open()))
        G = {g["condition_id"]: g for g in grows}
        crow = []
        for r in rows:
            g = G.get(r["condition_id"], {})
            x = {"condition_id": r["condition_id"], "cpu_status": r["status"], "gpu_status": g.get("status"),
                 "revision_same": g.get("revision") in (None, r["revision"])}
            for key in ("first_measure_peak_reserved", "first_measure_peak_allocated", "first_measure_start_reserved"):
                cv = r.get(key)
                gv = g.get(key) or g.get(f"summary.{key}")
                gv = int(float(gv)) if gv not in (None, "") else None
                x[f"cpu_{key}"], x[f"gpu_{key}"] = cv, gv
                x[f"{key}_diff_bytes"] = cv - gv if cv is not None and gv is not None else None
                x[f"{key}_rel"] = rel(cv, gv) if cv is not None and gv else None
            x["comparable"] = r["status"] == "ok" and str(g.get("status")) == "ok" and x["revision_same"]
            crow.append(x)
        write_csv(out / f"compare_{a.gpu_label or 'gpu'}.csv", crow)


if __name__ == "__main__":
    main()
