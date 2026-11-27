#!/usr/bin/env python3
"""Collect per-case result.json of a run into results/<run>/cpu_results.{jsonl,csv} + summary.json.

  python3 tools/collect.py --run-name pilot
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

EXP = Path("/home/inuos/cgf-fp16-680-20261111")
PRED = ["allocated_start_bytes", "allocated_peak_bytes", "allocated_end_bytes", "allocated_end_after_output_release_bytes",
        "reserved_start_bytes", "reserved_peak_bytes", "reserved_end_bytes", "reserved_end_after_output_release_bytes"]
TIMES = ["model_load", "input_prep", "trace_session_total", "warmup_runs_incl_profiler", "measure_run_incl_profiler",
         "trace_export", "trace_parse", "xmem_replay", "total_worker"]
STATUS_KO = {"ok": "성공", "nonfinite_output": "비유한 출력", "fp16_unsupported": "FP16 미지원", "input_mismatch": "입력 불일치",
             "cpu_ram_oom": "CPU RAM 부족", "timeout": "실행 제한 시간 초과", "unsupported_shape": "미지원 형태",
             "error": "기타 오류", "killed": "강제 종료", "no_result": "결과 없음", "replay_capacity_oom": "재생 용량 초과"}


def row_of(res, cdir):
    c = res.get("case", {})
    p = res.get("prediction") or {}
    t = res.get("timings_s") or {}
    rn = res.get("runner") or {}
    runs = res.get("runs") or []
    meas = [r for r in runs if r["run"].startswith("measure")]
    f = res.get("failure") or {}
    env = res.get("environment") or {}
    row = {"case_id": res.get("case_id"), "model": res.get("model") or c.get("model"), "revision": res.get("revision"),
           "kind": c.get("kind"), "dtype": c.get("dtype"), "batch_size": c.get("batch_size"), "input_len": c.get("input_len"),
           "new_tokens": c.get("new_tokens"), "resolution": c.get("resolution"), "pilot": c.get("pilot"),
           "status": res.get("status"), "status_ko": STATUS_KO.get(res.get("status"), res.get("status")),
           "failure_stage": f.get("stage"), "failure_run": f.get("run"), "failure_error": (f.get("error") or "")[:300],
           "actual_new_tokens_measure": meas[0]["actual_new_tokens_per_sample"] if meas else None,
           "actual_new_tokens_all_runs": ";".join(str(r["actual_new_tokens_per_sample"]) for r in runs) or None}
    for k, v in (res.get("inputs") or {}).items():
        row[f"input_{k}_shape"] = "x".join(map(str, v["shape"]))
        row[f"input_{k}_dtype"] = v["dtype"]
        row[f"input_{k}_sha256"] = v["sha256"]
    for k in PRED:
        row[f"cpu_{k}"] = p.get(k)
    rp = res.get("replay") or {}
    row["cpu_session_peak_allocated_bytes"] = rp.get("session_peak_allocated_bytes")
    row["cpu_session_peak_reserved_bytes"] = rp.get("session_peak_reserved_bytes")
    rs = res.get("resident") or {}
    row["resident_param_buffer_bytes"] = rs.get("parameter_buffer_bytes")
    row["resident_input_bytes"] = rs.get("input_bytes")
    ts = res.get("trace_stats") or {}
    row["trace_requests"] = ts.get("requests")
    row["trace_unmatched_frees"] = ts.get("unmatched_frees")
    row["trace_live_requests_at_end"] = ts.get("live_requests_at_session_end")
    for k in TIMES:
        row[f"time_{k}_s"] = round(t[k], 3) if isinstance(t.get(k), (int, float)) else None
    row["time_wall_container_s"] = round(rn["wall_seconds"], 3) if rn.get("wall_seconds") else None
    row["worker_max_rss_bytes"] = (res.get("rss") or {}).get("max_rss_bytes")
    row["container_memory_peak_bytes"] = rn.get("container_memory_peak_bytes")
    row["host_mem_available_before_bytes"] = (rn.get("host_mem_before") or {}).get("MemAvailable")
    row["threads"] = rn.get("threads") or env.get("torch_threads")
    row["mem_limit"] = rn.get("mem_limit")
    row["conv1d_colmajor_modules"] = (res.get("cpu_compute_modification") or {}).get("modules", 0)
    row["profiler_scope"] = (res.get("profiler") or {}).get("record_scopes")
    row["image_id"] = rn.get("image_id") or env.get("image_id")
    row["torch"], row["transformers"], row["python"] = env.get("torch"), env.get("transformers"), env.get("python")
    row["xmem_commit"] = (res.get("xmem") or {}).get("commit")
    row["finite_first_nonfinite"] = json.dumps((res.get("finite_check") or {}).get("first_nonfinite"))
    row["events_file"] = str(cdir / "events.npz") if (cdir / "events.npz").exists() else None
    row["result_file"] = str(cdir / "result.json")
    row["container_log"] = str(cdir / "container.log")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    a = ap.parse_args()
    root = EXP / "runs" / a.run_name / "cases"
    rows = []
    for cdir in sorted(p for p in root.iterdir() if p.is_dir() and ".attempt-" not in p.name):
        rp = cdir / "result.json"
        if rp.exists():
            rows.append(row_of(json.loads(rp.read_text()), cdir))
    out = EXP / "results" / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "cpu_results.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with (out / "cpu_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    summ = {"run": a.run_name, "cases_with_result": len(rows), "status": dict(Counter(r["status"] for r in rows)),
            "failures": [{k: r[k] for k in ("case_id", "status", "failure_stage", "failure_run", "failure_error")}
                         for r in rows if r["status"] != "ok"]}
    (out / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    print(json.dumps({k: summ[k] for k in ("run", "cases_with_result", "status")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
