#!/usr/bin/env python3
"""L256 CPU prediction scheduler (worker v2, unchanged). One fresh container per case.

Phase P (parallel, 3 slots by default)
  * every case to run goes through P, largest estimated memory first
  * slots have disjoint cpusets: large slot 0-12 (13 threads), small slots 13-17 / 18-22 (5 threads);
    one large case (estimate >= --large-gib) at a time, only in the large slot; sum of estimates <= --mem-budget-gib
  * memory watchdog (host MemAvailable, every 0.5 s): below --kill-below-gib with >= 2 cases running,
    the case with the SHORTEST elapsed time (most recently started) is killed -> status preempted_low_mem
  * a new case starts only when MemAvailable >= --admit-gib, >= --admit-after-kill-s after the last kill
    and >= --admit-gap-s after the last start
  * no new case below --min-disk-gib free disk (default 1.5)
  * per-container memory limit --par-mem (cgroup OOM -> worker_killed_cpu_oom); timeout --par-timeout-hours
Phase S (sequential, after P has completely finished)
  * cases whose P attempt ended by memory (preempted_low_mem / worker_killed_cpu_oom[_suspected] / cpu_oom) or timeout
  * cpus 0-19, 20 threads, --seq-mem, no timeout; the S result is final (no further retry)
Other failures are final in P. Finished cases (result.json with a final status, including reused symlinks) are skipped,
so the scheduler can be restarted.

  python3 tools/run_l256.py --run-name main395_l256 --cases manifest/cases395_l256.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run395 import DIAG, EXP, IMAGE, PREDICTOR, XMEM_WORK, cost, meminfo, poll_mem, sh  # noqa: E402

GiB = 1 << 30
MEMORY_FAIL = {"preempted_low_mem", "worker_killed_cpu_oom", "worker_killed_cpu_oom_suspected", "cpu_oom"}
DEFER = MEMORY_FAIL | {"timeout"}
LOCK = threading.Lock()
WORKER_ARGS: list = []


class Job:
    def __init__(self, case, phase, slot, cpuset, threads, mem, timeout_s):
        self.case, self.phase, self.slot, self.cpuset, self.threads, self.mem = case, phase, slot, cpuset, threads, mem
        self.timeout_s = timeout_s
        self.container = None
        self.start = None
        self.preempted = None
        self.est = 0.0


def log(path, rec):
    rec["t"] = time.time()
    with LOCK:
        with path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{time.strftime('%F %T')}] {json.dumps(rec)}", flush=True)


def run_job(job, run_root, cases_rel, running):
    c = job.case
    cdir = run_root / "cases" / c["case_id"]
    if cdir.is_symlink():
        raise RuntimeError(f"refusing to run a reused (symlinked) case: {cdir}")
    if cdir.exists():
        cdir.rename(cdir.with_name(f"{cdir.name}.attempt-{int(time.time())}"))
    cdir.mkdir(parents=True)
    image_id = sh("docker", "image", "inspect", IMAGE, "--format", "{{.Id}}").stdout.strip()
    name = f"cgfl256-{c['index']:03d}-{job.phase}-{int(time.time())}"
    cmd = ["docker", "run", "-d", "--name", name, "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}",
           "--memory", job.mem, "--memory-swap", job.mem, "--oom-score-adj", "500", "--cpuset-cpus", job.cpuset,
           "--tmpfs", "/tmp:exec,mode=1777",
           "-e", "HOME=/tmp", "-e", "MPLCONFIGDIR=/tmp", "-e", f"OMP_NUM_THREADS={job.threads}",
           "-e", f"MKL_NUM_THREADS={job.threads}", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "HF_HUB_OFFLINE=1",
           "-e", "TRANSFORMERS_OFFLINE=1", "-e", "HF_HOME=/exp/hf_cache", "-e", "PYTHONPATH=/exp/src:/diag/src:/predictor:/work/xMem",
           "-e", f"CGF_IMAGE={IMAGE}", "-e", f"CGF_IMAGE_ID={image_id}", "-e", f"CGF_MEM_LIMIT={job.mem}", "-e", f"CGF_CPUSET={job.cpuset}",
           "--mount", f"type=bind,src={XMEM_WORK},dst=/work,readonly",
           "--mount", f"type=bind,src={PREDICTOR},dst=/predictor,readonly",
           "--mount", f"type=bind,src={DIAG}/src,dst=/diag/src,readonly",
           "--mount", f"type=bind,src={EXP},dst=/exp",
           "-w", "/exp", "--entrypoint", "python", IMAGE, "-u", "-m", "cgf680.cpu_worker2",
           "--case-id", c["case_id"], "--cases", f"/exp/{cases_rel}", "--out", f"/exp/{cdir.relative_to(EXP)}",
           "--threads", str(job.threads)] + WORKER_ARGS
    before = meminfo()
    t0 = time.time()
    r = sh(*cmd)
    if r.returncode != 0:
        return {"case_id": c["case_id"], "status": "docker_run_failed", "phase": job.phase, "error": r.stderr[-1000:]}
    cid = r.stdout.strip()
    with LOCK:
        job.container, job.start = cid, t0
        running[c["case_id"]] = job
    box, stop = {}, threading.Event()
    th = threading.Thread(target=poll_mem, args=(cid, box, stop), daemon=True)
    th.start()
    logf = (cdir / "container.log").open("w")
    logp = subprocess.Popen(["docker", "logs", "-f", cid], stdout=logf, stderr=subprocess.STDOUT)
    timed_out = False
    try:
        w = sh("docker", "wait", cid, timeout=job.timeout_s)
        rc = int(w.stdout.strip() or -1)
    except subprocess.TimeoutExpired:
        timed_out = True
        sh("docker", "kill", cid)
        rc = int(sh("docker", "wait", cid).stdout.strip() or -1)
    wall = time.time() - t0
    with LOCK:
        running.pop(c["case_id"], None)
    stop.set()
    th.join(timeout=5)
    try:
        logp.wait(timeout=30)
    except subprocess.TimeoutExpired:
        logp.kill()
    logf.close()
    state = json.loads(sh("docker", "inspect", cid, "--format", "{{json .State}}").stdout or "{}")
    sh("docker", "rm", "-f", cid)
    runner = {"scheduler": "run_l256", "phase": job.phase, "slot": job.slot, "cpuset": job.cpuset, "threads": job.threads,
              "mem_limit": job.mem, "timeout_s": job.timeout_s, "exit_code": rc, "oom_killed": state.get("OOMKilled"),
              "timed_out": timed_out, "preempted": job.preempted, "wall_seconds": wall,
              "container_memory_peak_bytes": box.get("peak"), "container_memory_max_current_bytes": box.get("max_current"),
              "host_mem_before": before, "image": IMAGE, "image_id": image_id, "started": t0, "container": name,
              "cases_manifest": cases_rel}
    rp = cdir / "result.json"
    prog = json.loads((cdir / "progress.json").read_text()) if (cdir / "progress.json").exists() else {}
    res = json.loads(rp.read_text()) if rp.exists() else {"condition_id": c["case_id"], "case": c, "model": c["model"]}
    if job.preempted and (rc != 0 or res.get("status") != "ok"):   # a case that finished before the kill stays ok
        res["status"], res["failed_stage"] = "preempted_low_mem", prog.get("stage")
    elif timed_out:
        res["status"], res["failed_stage"] = "timeout", prog.get("stage")
    elif rc != 0 or not rp.exists() or res.get("status") in (None, "running"):
        if state.get("OOMKilled") or rc == 137:
            res["status"] = "worker_killed_cpu_oom" if state.get("OOMKilled") else "worker_killed_cpu_oom_suspected"
        else:
            res["status"] = f"worker_exit_{rc}"
        res["failed_stage"] = prog.get("stage")
    res["runner"] = runner
    rp.write_text(json.dumps(res, indent=1, default=str))
    return {"case_id": c["case_id"], "status": res["status"], "phase": job.phase, "slot": job.slot, "wall_s": round(wall, 1),
            "failed_stage": res.get("failed_stage"), "mem_peak": box.get("peak"),
            "first_measure_peak_reserved": (res.get("summary") or {}).get("first_measure_peak_reserved")}


def load_mem_estimates(run_names, cases, slack, oom_gib):
    """Peak container memory (GiB) per case, estimated from earlier runs of the same model/batch.
    Upper-bound neighbour: L or N == 256 -> 512 (L1024 table run). Memory failures -> oom_gib.
    Partial peaks of preempted attempts in the current run are lower bounds (max taken)."""
    att = {}
    for rn in run_names:
        f = EXP / "runs" / rn / "attempts.jsonl"
        if f.exists():
            for l in f.read_text().splitlines():
                r = json.loads(l)
                att.setdefault(rn, {})[r["case_id"]] = r
    est = {}
    for c in cases:
        cid = c["case_id"]
        e = None
        if c["kind"] == "vision_cls":
            up = cid   # vision conditions are identical across the L1024 and L256 tables
        else:
            up = f"{cid.split('__')[0]}__b{c['batch_size']}__in{512 if c['input_len'] == 256 else c['input_len']}__new{512 if c['new_tokens'] == 256 else c['new_tokens']}"
        if True:
            for rn in run_names:
                r = att.get(rn, {}).get(up) or att.get(rn, {}).get(cid)
                if r is None:
                    continue
                if r["status"] == "ok" and r.get("mem_peak"):
                    e = max(e or 0, r["mem_peak"] / GiB * slack + 1)
                elif r["status"] in MEMORY_FAIL - {"preempted_low_mem"}:
                    e = max(e or 0, oom_gib)
                elif r["status"] == "preempted_low_mem" and r.get("mem_peak"):
                    e = max(e or 0, r["mem_peak"] / GiB)
        est[cid] = e if e is not None else (4.0 if c["kind"] == "vision_cls" else oom_gib)
    return est


def status_of(run_root, cid):
    rp = run_root / "cases" / cid / "result.json"
    if not rp.exists():
        return None, None
    r = json.loads(rp.read_text())
    return r.get("status"), (r.get("runner") or {}).get("phase")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="main395_l256")
    ap.add_argument("--cases", default="manifest/cases395_l256.jsonl", help="path relative to the experiment root")
    ap.add_argument("--ids", nargs="*", default=None, help="restrict to these condition_ids")
    ap.add_argument("--cpusets", default="0-12,13-17,18-22",
                    help="parallel slots; the FIRST slot is the large slot (large cases run only there)")
    ap.add_argument("--par-threads", default="13,5,5", help="torch threads per slot (one value = all slots)")
    ap.add_argument("--par-mem", default="42g")
    ap.add_argument("--par-timeout-hours", type=float, default=8.0)
    ap.add_argument("--kill-below-gib", type=float, default=6.0)
    ap.add_argument("--kill-cooldown-s", type=float, default=2.0)
    ap.add_argument("--admit-gib", type=float, default=8.0)
    ap.add_argument("--mem-budget-gib", type=float, default=40.0, help="sum of estimated peaks of running cases")
    ap.add_argument("--large-gib", type=float, default=12.0, help="at most one running case at/above this estimate")
    ap.add_argument("--mem-estimates-from", nargs="*", default=["main395"], help="earlier run names (+ this run)")
    ap.add_argument("--mem-slack", type=float, default=1.15)
    ap.add_argument("--admit-after-kill-s", type=float, default=60.0)
    ap.add_argument("--admit-gap-s", type=float, default=30.0)
    ap.add_argument("--min-disk-gib", type=float, default=1.5, help="do not start a case below this free disk")
    ap.add_argument("--seq-cpus", default="0-19")
    ap.add_argument("--seq-threads", type=int, default=20)
    ap.add_argument("--seq-mem", default="42g")
    ap.add_argument("--phases", nargs="+", default=["P", "S"])
    ap.add_argument("--worker-args", default="", help="extra cpu_worker2 arguments, e.g. '--shadow-rule v2' (default none)")
    a = ap.parse_args()
    WORKER_ARGS[:] = a.worker_args.split()
    cases = [json.loads(l) for l in (EXP / a.cases).read_text().splitlines()]
    if a.ids is not None:
        cases = [c for c in cases if c["case_id"] in set(a.ids)]
    run_root = EXP / "runs" / a.run_name
    (run_root / "cases").mkdir(parents=True, exist_ok=True)
    attempts, events = run_root / "attempts.jsonl", run_root / "scheduler_events.jsonl"
    with (run_root / "suite_invocations.jsonl").open("a") as f:
        f.write(json.dumps({**vars(a), "n_cases": len(cases), "t": time.time()}) + "\n")

    par, seq = [], []
    for c in cases:
        st, ph = status_of(run_root, c["case_id"])
        if st is None:
            par.append(c)
        elif st in DEFER and ph == "P":
            seq.append(c)
    est = load_mem_estimates(list(a.mem_estimates_from) + [a.run_name], cases, a.mem_slack, 99.0)
    too_big = [c for c in par if est[c["case_id"]] > a.mem_budget_gib]
    par = [c for c in par if est[c["case_id"]] <= a.mem_budget_gib]
    seq += too_big
    for c in too_big:
        log(events, {"event": "defer_to_S_estimate", "case_id": c["case_id"], "est_gib": round(est[c["case_id"]], 1)})
    par.sort(key=lambda c: (est[c["case_id"]], cost(c)), reverse=True)
    print(f"to run in P: {len(par)} (large >= {a.large_gib} GiB: {sum(est[c['case_id']] >= a.large_gib for c in par)}), "
          f"deferred to S: {len(seq)} (by estimate {len(too_big)}), of {len(cases)}", flush=True)

    if "P" in a.phases and par:
        slots = a.cpusets.split(",")
        th = [int(x) for x in str(a.par_threads).split(",")]
        slot_threads = th * len(slots) if len(th) == 1 else th
        assert len(slot_threads) == len(slots), "--par-threads must have one value or one per slot"
        free = list(range(len(slots)))
        running, threads, pending = {}, [], []
        state = {"last_kill": 0.0, "last_start": 0.0, "stop": False}

        def watchdog():
            while not state["stop"]:
                avail = meminfo()["MemAvailable"]
                victim = None
                with LOCK:
                    live = [j for j in running.values() if j.container and not j.preempted]
                    if avail < a.kill_below_gib * GiB and len(live) >= 2:
                        victim = max(live, key=lambda j: j.start)
                        now = time.time()
                        victim.preempted = {"mem_available": avail, "elapsed_s": round(now - victim.start, 1),
                                            "running": {j.case["case_id"]: round(now - j.start, 1) for j in live}}
                if victim is not None:
                    log(events, {"event": "preempt", "case_id": victim.case["case_id"], **victim.preempted})
                    sh("docker", "kill", victim.container)
                    state["last_kill"] = time.time()
                    time.sleep(a.kill_cooldown_s)
                time.sleep(0.5)

        wd = threading.Thread(target=watchdog, daemon=True)
        wd.start()

        def slot_worker(job):
            try:
                rec = run_job(job, run_root, a.cases, running)
            except Exception as exc:  # noqa: BLE001
                rec = {"case_id": job.case["case_id"], "status": "scheduler_error", "phase": "P", "error": repr(exc)}
            log(attempts, rec)
            with LOCK:
                free.append(job.slot)
                if job in pending:
                    pending.remove(job)
                if rec["status"] in DEFER:
                    seq.append(job.case)

        queue = list(par)
        while queue or running or any(t.is_alive() for t in threads):
            time.sleep(1)
            if not queue:
                threads = [t for t in threads if t.is_alive()]
                continue
            now = time.time()
            with LOCK:
                can = bool(free) and now - state["last_kill"] >= a.admit_after_kill_s and now - state["last_start"] >= a.admit_gap_s
            if not can or meminfo()["MemAvailable"] < a.admit_gib * GiB:
                continue
            if shutil.disk_usage(EXP).free < a.min_disk_gib * GiB:
                if now - state.get("last_disk_log", 0) > 600:
                    state["last_disk_log"] = now
                    log(events, {"event": "wait_disk", "free_bytes": shutil.disk_usage(EXP).free})
                continue
            with LOCK:
                live = list(running.values()) + [j for j in pending if j.container is None]
                used = sum(j.est for j in live)
                large_running = any(j.est >= a.large_gib for j in live)
            large_left = any(est[q["case_id"]] >= a.large_gib for q in queue)
            with LOCK:
                free_now = sorted(free)
            pick = slot = None
            for i, c in enumerate(queue):   # largest first; one large case at a time (large slot 0), small ones fill
                e = est[c["case_id"]]
                if used + e > a.mem_budget_gib:
                    continue
                if e >= a.large_gib:
                    if large_running or 0 not in free_now:
                        continue
                    pick, slot = i, 0
                else:
                    small_slots = [k for k in free_now if k != 0]
                    if small_slots:
                        pick, slot = i, small_slots[0]
                    elif 0 in free_now and not large_left:   # large slot idle and no large case left
                        pick, slot = i, 0
                    else:
                        continue
                break
            if pick is None:
                continue
            c = queue.pop(pick)
            with LOCK:
                free.remove(slot)
            job = Job(c, "P", slot, slots[slot], slot_threads[slot], a.par_mem, a.par_timeout_hours * 3600)
            job.est = est[c["case_id"]]
            pending.append(job)
            state["last_start"] = now
            log(events, {"event": "start", "case_id": c["case_id"], "slot": slot, "mem_available": meminfo()["MemAvailable"],
                         "est_gib": round(job.est, 1), "running_est_gib": round(used, 1), "queue_left": len(queue)})
            t = threading.Thread(target=slot_worker, args=(job,))
            t.start()
            threads.append(t)
        state["stop"] = True
        print("PHASE_P_DONE", flush=True)

    if "S" in a.phases:
        seq_ids = []
        for c in sorted(seq, key=cost):
            if c["case_id"] in seq_ids:
                continue
            seq_ids.append(c["case_id"])
        print(f"phase S: {len(seq_ids)} {seq_ids}", flush=True)
        byid = {c["case_id"]: c for c in cases}
        for cid in seq_ids:
            while shutil.disk_usage(EXP).free < a.min_disk_gib * GiB:
                log(events, {"event": "wait_disk", "phase": "S", "free_bytes": shutil.disk_usage(EXP).free})
                time.sleep(600)
            job = Job(byid[cid], "S", None, a.seq_cpus, a.seq_threads, a.seq_mem, None)
            log(events, {"event": "start", "case_id": cid, "phase": "S", "mem_available": meminfo()["MemAvailable"]})
            rec = run_job(job, run_root, a.cases, {})
            log(attempts, rec)
        print("PHASE_S_DONE", flush=True)


if __name__ == "__main__":
    main()
