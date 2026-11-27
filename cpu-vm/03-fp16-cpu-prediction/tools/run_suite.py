#!/usr/bin/env python3
"""Sequential CPU-trace suite runner (host side). One fresh container per case.

  python3 tools/run_suite.py --run-name pilot --select pilot            # 26 pilot cases
  python3 tools/run_suite.py --run-name main  --select all              # 680 cases
  python3 tools/run_suite.py --run-name pilot --select ids --ids fp16__gpt2__b8__in128__new128

Resume: re-running the same --run-name skips cases whose result.json has a final status
(ok, nonfinite_output, fp16_unsupported, input_mismatch, unsupported_shape). cpu_ram_oom / timeout /
error / killed are skipped too unless --retry-failed. Every attempt is appended to attempts.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

EXP = Path("/home/inuos/cgf-fp16-680-20261111")
DIAG = Path("/home/inuos/cgf-reserved-diag-20261110")
XMEM_WORK = Path("/home/inuos/xmem-work")
PREDICTOR = Path("/home/inuos/CloudGPUFit-cpu-singlepass-20261108")
IMAGE = "cloudgpufit-cpu-validation:20261108"
FINAL = {"ok", "nonfinite_output", "fp16_unsupported", "input_mismatch", "unsupported_shape"}
RETRYABLE = {"cpu_ram_oom", "timeout", "error", "killed", "no_result", "replay_capacity_oom"}


def sh(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def meminfo():
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        if k in ("MemTotal", "MemAvailable", "MemFree"):
            out[k] = int(v.split()[0]) * 1024
    return out


def cgroup_dir(cid):
    for p in (f"/sys/fs/cgroup/system.slice/docker-{cid}.scope", f"/sys/fs/cgroup/docker/{cid}"):
        if Path(p).is_dir():
            return Path(p)
    return None


def poll_memory(cid, box, stop):
    d = None
    while not stop.is_set():
        if d is None:
            d = cgroup_dir(cid)
        if d is not None:
            try:
                cur = int((d / "memory.current").read_text())
                box["max_current"] = max(box.get("max_current", 0), cur)
                pk = d / "memory.peak"
                if pk.exists():
                    box["peak"] = int(pk.read_text())
            except (OSError, ValueError):
                pass
        stop.wait(1.0)


def run_case(case, a, run_root):
    cdir = run_root / "cases" / case["case_id"]
    res_path = cdir / "result.json"
    if res_path.exists():
        prev = json.loads(res_path.read_text()).get("status")
        if prev in FINAL or (prev in RETRYABLE and not a.retry_failed):
            return {"case_id": case["case_id"], "skipped": True, "status": prev}
        old = cdir.with_name(cdir.name + f".attempt-{int(time.time())}")
        cdir.rename(old)
    cdir.mkdir(parents=True, exist_ok=True)
    name = f"cgf680-{case['index']:03d}-{int(time.time())}"
    image_id = sh("docker", "image", "inspect", IMAGE, "--format", "{{.Id}}").stdout.strip()
    rel = cdir.relative_to(EXP)
    cmd = ["docker", "run", "-d", "--name", name, "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}",
           "--memory", a.mem, "--memory-swap", a.mem, "--cpus", str(a.threads),
           "--tmpfs", "/tmp:exec,mode=1777", "-e", "HOME=/tmp", "-e", "MPLCONFIGDIR=/tmp",
           "-e", f"OMP_NUM_THREADS={a.threads}", "-e", f"MKL_NUM_THREADS={a.threads}", "-e", "PYTHONDONTWRITEBYTECODE=1",
           "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1", "-e", "HF_HOME=/exp/hf_cache",
           "-e", "PYTHONPATH=/exp/src:/diag/src:/predictor:/work/xMem",
           "-e", f"CGF_IMAGE={IMAGE}", "-e", f"CGF_IMAGE_ID={image_id}", "-e", f"CGF_MEM_LIMIT={a.mem}",
           "-e", f"CGF_CPUS={a.threads}",
           "--mount", f"type=bind,src={XMEM_WORK},dst=/work,readonly",
           "--mount", f"type=bind,src={PREDICTOR},dst=/predictor,readonly",
           "--mount", f"type=bind,src={DIAG}/src,dst=/diag/src,readonly",
           "--mount", f"type=bind,src={EXP},dst=/exp",
           "-w", "/exp", "--entrypoint", "python", IMAGE,
           "-u", "-m", "cgf680.cpu_worker", "--case-id", case["case_id"], "--out", f"/exp/{rel}",
           "--cases", "/exp/" + str(Path(a.cases_file).resolve().relative_to(EXP)),
           "--threads", str(a.threads), "--profiler-scope", a.profiler_scope] + (["--conv1d-colmajor"] if a.conv1d_colmajor else []) + \
          (["--keep-raw-trace"] if a.keep_raw_trace else []) + \
          (["--emulate-fp16-from-fp32"] if a.emulate_fp16_from_fp32 else [])
    before = meminfo()
    t0 = time.time()
    r = sh(*cmd)
    if r.returncode != 0:
        return {"case_id": case["case_id"], "status": "docker_run_failed", "stderr": r.stderr[-2000:]}
    cid = r.stdout.strip()
    box, stop = {}, threading.Event()
    th = threading.Thread(target=poll_memory, args=(cid, box, stop), daemon=True)
    th.start()
    logf = (cdir / "container.log").open("w")
    logp = subprocess.Popen(["docker", "logs", "-f", cid], stdout=logf, stderr=subprocess.STDOUT)
    timed_out = False
    try:
        w = subprocess.run(["docker", "wait", cid], capture_output=True, text=True,
                           timeout=a.timeout_hours * 3600 if a.timeout_hours else None)
        exit_code = int(w.stdout.strip() or -1)
    except subprocess.TimeoutExpired:
        timed_out = True
        sh("docker", "kill", cid)
        exit_code = int(sh("docker", "wait", cid).stdout.strip() or -1)
    wall = time.time() - t0
    stop.set()
    th.join(timeout=5)
    logp.wait(timeout=30)
    logf.close()
    st = json.loads(sh("docker", "inspect", cid, "--format", "{{json .State}}").stdout or "{}")
    sh("docker", "rm", "-f", cid)
    runner = {"case_id": case["case_id"], "container": name, "exit_code": exit_code, "oom_killed": st.get("OOMKilled"),
              "timed_out": timed_out, "timeout_hours": a.timeout_hours, "wall_seconds": wall,
              "container_memory_peak_bytes": box.get("peak"), "container_memory_max_current_bytes": box.get("max_current"),
              "host_mem_before": before, "host_mem_after": meminfo(), "mem_limit": a.mem, "threads": a.threads,
              "image": IMAGE, "image_id": image_id, "conv1d_colmajor": a.conv1d_colmajor, "profiler_scope": a.profiler_scope,
              "emulate_fp16_from_fp32": a.emulate_fp16_from_fp32, "started": t0}
    (cdir / "runner.json").write_text(json.dumps(runner, indent=1))
    prog = json.loads((cdir / "progress.json").read_text()) if (cdir / "progress.json").exists() else {}
    if res_path.exists() and not timed_out and exit_code == 0:
        res = json.loads(res_path.read_text())
        status = res["status"]
    else:
        if timed_out:
            status = "timeout"
        elif st.get("OOMKilled") or exit_code == 137:
            status = "cpu_ram_oom" if st.get("OOMKilled") else "killed"
        else:
            status = json.loads(res_path.read_text())["status"] if res_path.exists() else "no_result"
        res = json.loads(res_path.read_text()) if res_path.exists() else {"case_id": case["case_id"], "case": case}
        if status in ("timeout", "cpu_ram_oom", "killed", "no_result"):
            res["status"] = status
            res["failure"] = {"stage": prog.get("stage"), "run": prog.get("run"),
                              "error": f"exit={exit_code} oom_killed={st.get('OOMKilled')} timed_out={timed_out}"}
            res["model"], res["revision"] = case["model"], json.loads((EXP / "manifest/model_pins.json").read_text())[case["model"]]["revision"]
            res_path.write_text(json.dumps(res, indent=1))
    res["runner"] = runner
    res_path.write_text(json.dumps(res, indent=1, default=str))
    return {"case_id": case["case_id"], "status": status, "wall_seconds": round(wall, 1), "stage": prog.get("stage"),
            "run": prog.get("run"), "peak_mem": box.get("peak"), "prediction": res.get("prediction")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--select", choices=["pilot", "all", "ids"], required=True)
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--models", nargs="*", default=[], help="filter by short model name")
    ap.add_argument("--order", choices=["manifest", "cost"], default="cost")
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--mem", default="42g")
    ap.add_argument("--timeout-hours", type=float, default=None)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--conv1d-colmajor", action="store_true")
    ap.add_argument("--emulate-fp16-from-fp32", action="store_true")
    ap.add_argument("--keep-raw-trace", action="store_true")
    ap.add_argument("--profiler-scope", choices=["user", "all"], default="user")
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--cases-file", default=str(EXP / "manifest/cases.jsonl"))
    a = ap.parse_args()
    cases = [json.loads(l) for l in Path(a.cases_file).read_text().splitlines()]
    if a.select == "pilot":
        cases = [c for c in cases if c["pilot"]]
    elif a.select == "ids":
        want = set(a.ids)
        cases = [c for c in cases if c["case_id"] in want]
        if not cases:
            sys.exit("no matching case ids")
    if a.models:
        cases = [c for c in cases if c["short"] in a.models]
    if a.order == "cost" and a.select != "ids":  # rough work proxy: small models / small shapes first
        size_rank = {"opt-125m": 1, "gpt2": 1, "t5-small": 1, "resnet-18": 0, "mobilenet-v2": 0, "resnet-50": 0,
                     "gpt2-medium": 3, "t5-base": 2, "vit-large": 2, "tinyllama-1.1b": 8, "qwen2.5-1.5b": 10,
                     "gpt2-xl": 15, "t5-3b": 25}
        cases.sort(key=lambda c: size_rank[c["short"]] * c["batch_size"] *
                   ((c["input_len"] or 0) + (c["new_tokens"] or 0) + ((c["resolution"] or 0) ** 2) / 256 + 1))
    if a.max_cases:
        cases = cases[:a.max_cases]
    run_root = EXP / "runs" / a.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "suite_args.json").write_text(json.dumps({**vars(a), "n_cases": len(cases), "started": time.time()}, indent=1))
    for i, c in enumerate(cases):
        print(f"[{time.strftime('%F %T')}] {i + 1}/{len(cases)} {c['case_id']} ...", flush=True)
        rec = run_case(c, a, run_root)
        rec["t"] = time.time()
        with (run_root / "attempts.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{time.strftime('%F %T')}]   -> {json.dumps(rec)}", flush=True)
    print("SUITE_DONE", flush=True)


if __name__ == "__main__":
    main()
