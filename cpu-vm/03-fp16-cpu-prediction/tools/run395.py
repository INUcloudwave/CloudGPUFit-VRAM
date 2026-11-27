#!/usr/bin/env python3
"""395-condition CPU prediction scheduler (worker v2). One fresh container per case.

Phases (same run directory; finished cases are never re-run):
  A  parallel pool : "small" cases, 4 slots x 5 pinned CPUs (disjoint cpusets), 10g RAM each, timeout 2 h
  B  sequential    : all other cases, 20 CPUs, 42g RAM, timeout 2 h
  C  retry         : (A cases killed by OOM / cpu_oom) + (A/B timeouts) -> sequential, 20 CPUs, 42g, NO timeout
                     (OOM in a sequential run is final; it is not retried)
Usage:
  python3 tools/run395.py --run-name main395 --select pilot --phases A B      # pilot (26)
  python3 tools/run395.py --run-name main395 --select all   --phases A B C    # remaining 369 + retries
Cases are independent: xMem predictions depend only on the request trace, not on timing; parallel slots share
no CPUs; RAM limits are per container (sum 40g < host 47g); phase B never overlaps phase A.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
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
SLOTS = ["0-4", "5-9", "10-14", "15-19"]
SEQ_CPUS = "0-19"
SMALL_MODELS = {"opt-125m", "gpt2", "gpt2-medium", "t5-small", "t5-base", "resnet-18", "resnet-50",
                "mobilenet-v2-1.0-224", "vit-large-patch16-224"}
LOCK = threading.Lock()


def sh(*cmd, timeout=None):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def is_small(c):
    if c["short"] not in SMALL_MODELS:
        return False
    b = c["batch_size"]
    if c["kind"] == "vision_cls":
        return b * c["resolution"] ** 2 <= 32 * 256 ** 2
    tok = b * (c["input_len"] + c["new_tokens"])
    limit = 8 * 1024 if c["short"] in ("gpt2-medium", "t5-base") else 16 * 1024
    return tok <= limit


def cost(c):
    rank = {"resnet-18": 0.2, "mobilenet-v2-1.0-224": 0.2, "resnet-50": 0.4, "vit-large-patch16-224": 0.5, "opt-125m": 1,
            "gpt2": 1, "t5-small": 1, "gpt2-medium": 3, "t5-base": 2, "tinyllama-1.1b-chat": 8, "qwen2.5-1.5b-instruct": 10,
            "gpt2-xl": 15, "t5-3b": 25}[c["short"]]
    if c["kind"] == "vision_cls":
        return rank * c["batch_size"] * c["resolution"] ** 2 / 256
    return rank * c["batch_size"] * (c["input_len"] + c["new_tokens"])


def meminfo():
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        if k in ("MemTotal", "MemAvailable"):
            out[k] = int(v.split()[0]) * 1024
    return out


def poll_mem(cid, box, stop):
    d = None
    while not stop.is_set():
        if d is None:
            for p in (f"/sys/fs/cgroup/system.slice/docker-{cid}.scope", f"/sys/fs/cgroup/docker/{cid}"):
                if Path(p).is_dir():
                    d = Path(p)
        if d is not None:
            try:
                box["max_current"] = max(box.get("max_current", 0), int((d / "memory.current").read_text()))
                if (d / "memory.peak").exists():
                    box["peak"] = int((d / "memory.peak").read_text())
            except (OSError, ValueError):
                pass
        stop.wait(1.0)


def run_one(c, run_root, phase, cpuset, threads, mem, timeout_s, attempt):
    cdir = run_root / "cases" / c["case_id"]
    if cdir.exists():
        cdir.rename(cdir.with_name(f"{cdir.name}.attempt-{int(time.time())}"))
    cdir.mkdir(parents=True)
    image_id = sh("docker", "image", "inspect", IMAGE, "--format", "{{.Id}}").stdout.strip()
    name = f"cgf395-{c['index']:03d}-{phase}-{int(time.time())}"
    cmd = ["docker", "run", "-d", "--name", name, "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}",
           "--memory", mem, "--memory-swap", mem, "--cpuset-cpus", cpuset, "--tmpfs", "/tmp:exec,mode=1777",
           "-e", "HOME=/tmp", "-e", "MPLCONFIGDIR=/tmp", "-e", f"OMP_NUM_THREADS={threads}", "-e", f"MKL_NUM_THREADS={threads}",
           "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
           "-e", "HF_HOME=/exp/hf_cache", "-e", "PYTHONPATH=/exp/src:/diag/src:/predictor:/work/xMem",
           "-e", f"CGF_IMAGE={IMAGE}", "-e", f"CGF_IMAGE_ID={image_id}", "-e", f"CGF_MEM_LIMIT={mem}", "-e", f"CGF_CPUSET={cpuset}",
           "--mount", f"type=bind,src={XMEM_WORK},dst=/work,readonly",
           "--mount", f"type=bind,src={PREDICTOR},dst=/predictor,readonly",
           "--mount", f"type=bind,src={DIAG}/src,dst=/diag/src,readonly",
           "--mount", f"type=bind,src={EXP},dst=/exp",
           "-w", "/exp", "--entrypoint", "python", IMAGE, "-u", "-m", "cgf680.cpu_worker2",
           "--case-id", c["case_id"], "--out", f"/exp/{cdir.relative_to(EXP)}", "--threads", str(threads)]
    before = meminfo()
    t0 = time.time()
    r = sh(*cmd)
    if r.returncode != 0:
        return {"case_id": c["case_id"], "status": "docker_run_failed", "phase": phase, "error": r.stderr[-1000:]}
    cid = r.stdout.strip()
    box, stop = {}, threading.Event()
    th = threading.Thread(target=poll_mem, args=(cid, box, stop), daemon=True)
    th.start()
    logf = (cdir / "container.log").open("w")
    logp = subprocess.Popen(["docker", "logs", "-f", cid], stdout=logf, stderr=subprocess.STDOUT)
    timed_out = False
    try:
        w = sh("docker", "wait", cid, timeout=timeout_s)
        rc = int(w.stdout.strip() or -1)
    except subprocess.TimeoutExpired:
        timed_out = True
        sh("docker", "kill", cid)
        rc = int(sh("docker", "wait", cid).stdout.strip() or -1)
    wall = time.time() - t0
    stop.set()
    th.join(timeout=5)
    try:
        logp.wait(timeout=30)
    except subprocess.TimeoutExpired:
        logp.kill()
    logf.close()
    state = json.loads(sh("docker", "inspect", cid, "--format", "{{json .State}}").stdout or "{}")
    sh("docker", "rm", "-f", cid)
    runner = {"phase": phase, "attempt": attempt, "cpuset": cpuset, "threads": threads, "mem_limit": mem,
              "timeout_s": timeout_s, "exit_code": rc, "oom_killed": state.get("OOMKilled"), "timed_out": timed_out,
              "wall_seconds": wall, "container_memory_peak_bytes": box.get("peak"),
              "container_memory_max_current_bytes": box.get("max_current"), "host_mem_before": before,
              "image": IMAGE, "image_id": image_id, "started": t0, "container": name}
    rp = cdir / "result.json"
    prog = json.loads((cdir / "progress.json").read_text()) if (cdir / "progress.json").exists() else {}
    res = json.loads(rp.read_text()) if rp.exists() else {"condition_id": c["case_id"], "case": c, "model": c["model"]}
    if timed_out:
        res["status"], res["failed_stage"] = "timeout", prog.get("stage")
    elif rc != 0 or not rp.exists() or res.get("status") in (None, "running"):
        if state.get("OOMKilled") or rc == 137:
            res["status"] = "worker_killed_cpu_oom" if state.get("OOMKilled") else "worker_killed_cpu_oom_suspected"
        else:
            res["status"] = f"worker_exit_{rc}"
        res["failed_stage"] = prog.get("stage")
    res["runner"] = runner
    rp.write_text(json.dumps(res, indent=1, default=str))
    return {"case_id": c["case_id"], "status": res["status"], "phase": phase, "attempt": attempt, "wall_s": round(wall, 1),
            "failed_stage": res.get("failed_stage"), "mem_peak": box.get("peak"),
            "first_measure_peak_reserved": (res.get("summary") or {}).get("first_measure_peak_reserved")}


def log(run_root, rec):
    rec["t"] = time.time()
    with LOCK:
        with (run_root / "attempts.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{time.strftime('%F %T')}] {json.dumps(rec)}", flush=True)


def current_status(run_root, case_id):
    rp = run_root / "cases" / case_id / "result.json"
    if not rp.exists():
        return None, None
    r = json.loads(rp.read_text())
    return r.get("status"), (r.get("runner") or {}).get("phase")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="main395")
    ap.add_argument("--select", choices=["pilot", "all", "ids"], default="all")
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--phases", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--timeout-hours", type=float, default=2.0)
    a = ap.parse_args()
    cases = [json.loads(l) for l in (EXP / "manifest/cases395.jsonl").read_text().splitlines()]
    if a.select == "pilot":
        cases = [c for c in cases if c["pilot"]]
    elif a.select == "ids":
        cases = [c for c in cases if c["case_id"] in set(a.ids)]
    run_root = EXP / "runs" / a.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / "suite_invocations.jsonl").open("a") as f:
        f.write(json.dumps({**vars(a), "n_cases": len(cases), "t": time.time()}) + "\n")
    todo = [c for c in cases if current_status(run_root, c["case_id"])[0] is None]
    small = sorted([c for c in todo if is_small(c)], key=cost)
    large = sorted([c for c in todo if not is_small(c)], key=cost)
    tmo = a.timeout_hours * 3600
    print(f"todo {len(todo)} (small {len(small)}, large {len(large)}) of {len(cases)}", flush=True)
    if "A" in a.phases and small:
        q = queue.Queue()
        for c in small:
            q.put(c)

        def worker(cpuset):
            while True:
                try:
                    c = q.get_nowait()
                except queue.Empty:
                    return
                log(run_root, run_one(c, run_root, "A", cpuset, 5, "10g", tmo, 1))
        ths = [threading.Thread(target=worker, args=(s,)) for s in SLOTS]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        print("PHASE_A_DONE", flush=True)
    if "B" in a.phases:
        for c in large:
            if current_status(run_root, c["case_id"])[0] is None:
                log(run_root, run_one(c, run_root, "B", SEQ_CPUS, 20, "42g", tmo, 1))
        print("PHASE_B_DONE", flush=True)
    if "C" in a.phases:
        retry = []
        for c in cases:
            st, ph = current_status(run_root, c["case_id"])
            if st == "timeout" and ph in ("A", "B"):
                retry.append(c)
            elif st in ("worker_killed_cpu_oom", "worker_killed_cpu_oom_suspected", "cpu_oom") and ph == "A":
                retry.append(c)
        print(f"retry {len(retry)}: {[c['case_id'] for c in retry]}", flush=True)
        for c in sorted(retry, key=cost):
            log(run_root, run_one(c, run_root, "C", SEQ_CPUS, 20, "42g", None, 2))
        print("PHASE_C_DONE", flush=True)
    print("SUITE_DONE", flush=True)


if __name__ == "__main__":
    main()
