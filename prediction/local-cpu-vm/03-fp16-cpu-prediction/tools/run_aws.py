#!/usr/bin/env python3
"""AWS CPU scheduler for the 395-condition L256 table (cpu_worker2, one fresh container per condition).

Same image, worker, protocol, condition table and per-condition torch thread count as the INUOS run (main395_l256);
only the packing is changed so that many conditions run at once without affecting each other's results:

  * torch threads = the INUOS value of that condition (5 / 13 / 20, from aws/inuos_case_stats.json).
    The request stream depends on the thread count (CPU kernels allocate per-thread scratch), NOT on how many cores
    the container gets: verified on INUOS (gpt2 b1, 5 threads on 1 core == stored 5 threads on 5 cores, byte-equal).
  * cores per container (disjoint --cpuset-cpus) from the INUOS wall time: most of a run is single-threaded Python
    (profiler bookkeeping, matching, xMem replay), so small conditions get 1 core, large ones threads x trace share x 0.5.
  * memory limit per container = 1.6 x INUOS peak + 2 GiB (min 6 GiB); admission keeps the sum of limits below
    host RAM - reserve, so no container can starve another (each has its own cgroup limit).
  * order: longest INUOS wall first (LPT) in phase P. Conditions that failed on INUOS (cpu OOM) and conditions that
    fail here by memory/timeout/docker error go to phase S, which starts after P has finished: bigger memory, more cores.
  * resumable: a condition whose result.json has a final status is skipped.

  python3 tools/run_aws.py --run-name aws_fp16_l256 --worker-args "--dtype-mode fp16_real"
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

GiB = 1 << 30
EXP = Path(__file__).resolve().parents[1]
FINAL = {"ok", "invalid_input", "generation_length_mismatch", "fp16_unsupported", "compat_or_runtime_error"}
RETRY = {"worker_killed_cpu_oom", "worker_killed_cpu_oom_suspected", "cpu_oom", "timeout", "docker_run_failed",
         "no_result"}
LOCK = threading.Lock()


def sh(*cmd, timeout=None):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def meminfo():
    d = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":")
        d[k] = int(v.split()[0]) * 1024
    return d


def cores_for(st, k=0.5):
    """Cores for one container: the multi-threaded part of a run is the traced invocation (INUOS trace_s); the rest is
    single-threaded. cores ~ threads x (trace share of the INUOS wall) x k, at least 2 for runs >= 5 min, 1 below.
    (Simulated on the INUOS stats for 64 cores: phase P 3.7-6.5 h vs 3.9-9.4 h for a wall-only rule.)"""
    w = st["wall_s"]
    if w < 300:
        return 1
    f = st["trace_s"] / max(w, 1.0)
    return int(min(st["threads"], max(2, round(st["threads"] * f * k))))


def mem_for(st):
    return max(6, math.ceil(1.6 * st["mem_peak_gib"] + 2))


class Pool:
    """Disjoint cores + memory budget. take() is non-blocking (dispatcher decides order)."""

    def __init__(self, ncpu, mem_bytes):
        self.free = list(range(ncpu))
        self.mem_free = mem_bytes
        self.cv = threading.Condition()

    def try_take(self, n, mem):
        if len(self.free) < n or self.mem_free < mem:
            return None
        cpus = sorted(self.free)[:n]
        for c in cpus:
            self.free.remove(c)
        self.mem_free -= mem
        return cpus

    def give(self, cpus, mem):
        with self.cv:
            self.free += cpus
            self.mem_free += mem
            self.cv.notify_all()


def dispatch(pool, queue, need, run, min_disk_bytes):
    """queue in priority order (longest first). Start the first queued item that fits the free cores/memory
    (large items first while resources are free, small items backfill). Returns when everything has finished."""
    threads = []
    queue = list(queue)
    while queue:
        with pool.cv:
            started = False
            if shutil.disk_usage(EXP).free >= min_disk_bytes:
                for k, c in enumerate(queue):
                    n, mem = need(c)
                    cpus = pool.try_take(n, mem)
                    if cpus is not None:
                        queue.pop(k)
                        th = threading.Thread(target=run, args=(c, cpus, n, mem), daemon=True)
                        th.start()
                        threads.append(th)
                        started = True
                        break
            if not started:
                pool.cv.wait(10)
        if started:
            time.sleep(2)                      # stagger model loading
    for th in threads:
        th.join()


def log(path, ev):
    ev["t"] = time.time()
    line = json.dumps(ev)
    with LOCK:
        with open(path, "a") as f:
            f.write(line + "\n")
    print(time.strftime("[%H:%M:%S] ") + line, flush=True)


def final_status(cdir):
    try:
        return json.loads((cdir / "result.json").read_text()).get("status")
    except Exception:
        return None


def run_one(a, case, phase, threads, cpus, mem_gib, timeout_s, run_root, events):
    cdir = run_root / "cases" / case["case_id"]
    if cdir.exists():
        cdir.rename(cdir.with_name(f"{cdir.name}.attempt-{int(time.time())}"))
    cdir.mkdir(parents=True)
    cpuset = ",".join(map(str, cpus))
    mem = f"{mem_gib}g"
    name = f"cgfaws-{case['index']:03d}-{phase}-{int(time.time() * 1000) % 10**9}"
    cmd = ["docker", "run", "-d", "--name", name, "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}",
           "--memory", mem, "--memory-swap", mem, "--oom-score-adj", "500", "--cpuset-cpus", cpuset,
           "--tmpfs", "/tmp:exec,mode=1777",
           "-e", "HOME=/tmp", "-e", "MPLCONFIGDIR=/tmp", "-e", f"OMP_NUM_THREADS={threads}",
           "-e", f"MKL_NUM_THREADS={threads}", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "HF_HUB_OFFLINE=1",
           "-e", "TRANSFORMERS_OFFLINE=1", "-e", "HF_HOME=/exp/hf_cache",
           "-e", "PYTHONPATH=/exp/src:/diag/src:/predictor:/work/xMem",
           "-e", f"CGF_IMAGE={a.image}", "-e", f"CGF_IMAGE_ID={a.image_id}", "-e", f"CGF_MEM_LIMIT={mem}",
           "-e", f"CGF_CPUSET={cpuset}",
           "--mount", f"type=bind,src={a.xmem_work},dst=/work,readonly",
           "--mount", f"type=bind,src={a.predictor},dst=/predictor,readonly",
           "--mount", f"type=bind,src={a.diag_src},dst=/diag/src,readonly",
           "--mount", f"type=bind,src={EXP},dst=/exp",
           "-w", "/exp", "--entrypoint", "python", a.image, "-u", "-m", "cgf680.cpu_worker2",
           "--case-id", case["case_id"], "--cases", f"/exp/{a.cases}", "--out", f"/exp/{cdir.relative_to(EXP)}",
           "--threads", str(threads)] + a.worker_args.split()
    t0 = time.time()
    r = sh(*cmd)
    if r.returncode != 0:
        return {"status": "docker_run_failed", "error": r.stderr[-800:], "wall": 0}
    cid = r.stdout.strip()
    logf = (cdir / "container.log").open("w")
    lp = subprocess.Popen(["docker", "logs", "-f", cid], stdout=logf, stderr=subprocess.STDOUT)
    timed_out = False
    try:
        w = sh("docker", "wait", cid, timeout=timeout_s)
        code = int(w.stdout.strip() or -1)
    except subprocess.TimeoutExpired:
        timed_out = True
        sh("docker", "kill", cid)
        code = -9
    insp = sh("docker", "inspect", cid, "--format", "{{.State.OOMKilled}}").stdout.strip()
    lp.terminate()
    logf.close()
    sh("docker", "rm", "-f", cid)
    st = final_status(cdir)
    if timed_out:
        st = "timeout"
    elif st in (None, "running"):
        st = "worker_killed_cpu_oom" if insp == "true" or code == 137 else "no_result"
    runner = {"scheduler": "run_aws", "phase": phase, "threads": threads, "cpuset": cpuset, "cores": len(cpus),
              "mem_limit": mem, "exit_code": code, "oom_killed": insp == "true", "timed_out": timed_out,
              "wall_seconds": time.time() - t0, "image": a.image, "image_id": a.image_id, "container": name,
              "worker_args": a.worker_args}
    try:
        res = json.loads((cdir / "result.json").read_text())
        res["runner"] = runner
        if st != res.get("status"):
            res["status"] = st
        (cdir / "result.json").write_text(json.dumps(res, indent=1, default=str))
    except Exception:
        (cdir / "result.json").write_text(json.dumps({"condition_id": case["case_id"], "status": st, "runner": runner}))
    return {"status": st, "wall": runner["wall_seconds"], "exit_code": code}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--cases", default="manifest/cases395_l256.jsonl")
    ap.add_argument("--stats", default="aws/inuos_case_stats.json")
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--worker-args", default="--dtype-mode fp16_real")
    ap.add_argument("--image", default="cloudgpufit-cpu-validation:20261108")
    ap.add_argument("--xmem-work", default=str(EXP.parent / "xmem-work"))
    ap.add_argument("--predictor", default=str(EXP.parent / "predictor"))
    ap.add_argument("--diag-src", default=str(EXP.parent / "diag_src"))
    ap.add_argument("--cpus", type=int, default=os.cpu_count(), help="cores to schedule on")
    ap.add_argument("--reserve-gib", type=float, default=12.0, help="RAM kept free for the host")
    ap.add_argument("--min-disk-gib", type=float, default=30.0)
    ap.add_argument("--timeout-factor", type=float, default=6.0, help="P timeout = max(2 h, factor x INUOS wall)")
    ap.add_argument("--s-mem-gib", type=int, default=96, help="phase S memory per container (INUOS FP32 OOM was at 42 GiB)")
    ap.add_argument("--s-cores", type=int, default=12)
    ap.add_argument("--phases", nargs="+", default=["P", "S"])
    a = ap.parse_args()
    a.image_id = sh("docker", "image", "inspect", a.image, "--format", "{{.Id}}").stdout.strip()
    if not a.image_id:
        raise SystemExit(f"image {a.image} not loaded")
    stats = json.loads((EXP / a.stats).read_text())["cases"]
    cases = [json.loads(x) for x in (EXP / a.cases).read_text().splitlines()]
    if a.ids:
        cases = [c for c in cases if c["case_id"] in set(a.ids)]
    run_root = EXP / "runs" / a.run_name
    (run_root / "cases").mkdir(parents=True, exist_ok=True)
    events = run_root / "scheduler_events.jsonl"
    total = meminfo()["MemTotal"]
    pool_mem = int(total - a.reserve_gib * GiB)
    pool = Pool(a.cpus, pool_mem)
    s_mem = min(a.s_mem_gib, int(pool_mem / GiB))
    log(events, {"event": "init", "cpus": a.cpus, "mem_total_gib": round(total / GiB, 1), "pool_mem_gib": round(pool_mem / GiB, 1),
                 "worker_args": a.worker_args, "image_id": a.image_id, "n_cases": len(cases)})

    def done(c):
        return final_status(run_root / "cases" / c["case_id"]) in FINAL

    todo = [c for c in cases if not done(c)]
    p_list = [c for c in todo if stats[c["case_id"]]["status"] == "ok"]
    s_list = [c for c in todo if stats[c["case_id"]]["status"] != "ok"]
    p_list.sort(key=lambda c: -stats[c["case_id"]]["wall_s"])
    retry = []

    def need_p(c):
        st = stats[c["case_id"]]
        return min(cores_for(st), a.cpus), mem_for(st) * GiB

    def run_p(c, cpus, n, mem_b):
        st = stats[c["case_id"]]
        mem = mem_b // GiB
        timeout = max(7200.0, a.timeout_factor * st["wall_s"])
        log(events, {"event": "start", "phase": "P", "case_id": c["case_id"], "threads": st["threads"], "cores": n,
                     "mem_gib": mem, "inuos_wall_s": st["wall_s"]})
        try:
            r = run_one(a, c, "P", st["threads"], cpus, mem, timeout, run_root, events)
        finally:
            pool.give(cpus, mem_b)
        log(events, {"event": "end", "phase": "P", "case_id": c["case_id"], **r})
        if r["status"] in RETRY:
            with LOCK:
                retry.append(c)

    if "P" in a.phases and p_list:
        log(events, {"event": "phase_P", "n": len(p_list)})
        dispatch(pool, p_list, need_p, run_p, a.min_disk_gib * GiB)
    if "S" in a.phases:
        s_all = s_list + [c for c in retry if c not in s_list]
        s_all.sort(key=lambda c: -stats[c["case_id"]]["wall_s"])
        log(events, {"event": "phase_S", "n": len(s_all), "mem_gib": s_mem, "cores": a.s_cores})

        def need_s(c):
            return min(a.s_cores, a.cpus), s_mem * GiB

        def run_s(c, cpus, n, mem_b):
            th = stats[c["case_id"]]["threads"]
            log(events, {"event": "start", "phase": "S", "case_id": c["case_id"], "threads": th, "cores": n, "mem_gib": s_mem})
            try:
                r = run_one(a, c, "S", th, cpus, s_mem, 48 * 3600, run_root, events)
            finally:
                pool.give(cpus, mem_b)
            log(events, {"event": "end", "phase": "S", "case_id": c["case_id"], **r})

        dispatch(pool, s_all, need_s, run_s, a.min_disk_gib * GiB)
    final = {c["case_id"]: final_status(run_root / "cases" / c["case_id"]) for c in cases}
    from collections import Counter
    log(events, {"event": "done", "status_counts": dict(Counter(final.values()))})
    print("RUN_AWS_DONE", flush=True)


if __name__ == "__main__":
    main()
