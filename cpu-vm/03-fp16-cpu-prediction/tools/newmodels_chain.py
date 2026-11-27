#!/usr/bin/env python3
"""Run chain for the generalisation test (2026-11-22, user instruction):
  1. wait until the running large case finishes (its measured memory calibrates the estimates)
  2. stop the first scheduler (its tmux session and its containers cgfl256-*; nothing else) - unfinished cases rerun
  3. new memory estimates: mapped old-model peak x (measured new/old ratio of the same model, max over finished cases)
  4. main run again, phase P only, up to 3 cases in parallel (no "one large case at a time" rule)
  5. cases that failed by memory/timeout: retry 2 in parallel (separate run dir *_retry2, P), then the ones that fail
     again alone (S of the same retry run, 20 threads, 42 GiB)
usage: python3 tools/newmodels_chain.py <case_id to wait for>
"""
import json
import subprocess
import sys
import time
from pathlib import Path

E = Path(__file__).resolve().parents[1]
RUN = "newmodels_20261122"
CASES = "manifest/cases_newmodels_20261122.jsonl"
WA = "--models /exp/manifest/models_newmodels_20261122.json"
MAP = {"convnext-tiny-224": "resnet-50", "efficientnet-b0": "mobilenet-v2-1.0-224", "regnet-y-040": "resnet-50",
       "qwen2.5-0.5b-instruct": "qwen2.5-1.5b-instruct", "smollm2-360m-instruct": "tinyllama-1.1b-chat",
       "pythia-410m-deduped": "tinyllama-1.1b-chat"}
DEFER = {"preempted_low_mem", "worker_killed_cpu_oom", "worker_killed_cpu_oom_suspected", "cpu_oom", "timeout"}
GiB = 1 << 30


def log(msg):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg, flush=True)


def status(run, cid):
    p = E / "runs" / run / "cases" / cid / "result.json"
    return json.loads(p.read_text()).get("status") if p.exists() else None


def old_peaks():
    out = {}
    for rn in ("main395", "main395_l256"):
        f = E / "runs" / rn / "attempts.jsonl"
        if f.exists():
            for l in f.read_text().splitlines():
                r = json.loads(l)
                if r.get("status") == "ok" and r.get("mem_peak"):
                    out[r["case_id"]] = max(out.get(r["case_id"], 0), r["mem_peak"])
    return out


def main():
    wait_for = sys.argv[1]
    cases = [json.loads(l) for l in (E / CASES).read_text().splitlines()]
    while status(RUN, wait_for) is None:
        time.sleep(30)
    log(f"{wait_for}: {status(RUN, wait_for)}")
    subprocess.run(["tmux", "kill-session", "-t", "cgf-new"], check=False)
    names = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True).stdout.split()
    victims = [n for n in names if n.startswith("cgfl256-")]
    if victims:
        subprocess.run(["docker", "kill"] + victims, check=False, capture_output=True)
    log(f"stopped first scheduler, killed containers {victims}")
    # calibrated estimates
    old = old_peaks()
    new = {}
    f = E / "runs" / RUN / "attempts.jsonl"
    for l in f.read_text().splitlines():
        r = json.loads(l)
        if r.get("status") == "ok" and r.get("mem_peak"):
            new[r["case_id"]] = max(new.get(r["case_id"], 0), r["mem_peak"])
    ratio = {}
    for c in cases:
        o = old.get(MAP[c["short"]] + "__" + c["case_id"].split("__", 1)[1])
        if c["case_id"] in new and o:
            ratio[c["short"]] = max(ratio.get(c["short"], 0), new[c["case_id"]] / o)
    log(f"ratio new/old peak per model: { {k: round(v, 2) for k, v in ratio.items()} }")
    d = E / "runs" / "newmodels_mem_estimates_v2"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in cases:
        o = old.get(MAP[c["short"]] + "__" + c["case_id"].split("__", 1)[1], 30 * GiB)
        rows.append({"case_id": c["case_id"], "status": "ok", "mem_peak": int(new.get(c["case_id"]) or o * ratio.get(c["short"], 1.0)),
                     "note": "calibrated estimate (newmodels_chain.py)"})
    (d / "attempts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    # main run, P only, 3 parallel (large rule off)
    base = ["python3", "tools/run_newmodels.py", "--cases", CASES, "--worker-args", WA, "--min-disk-gib", "3"]
    log("main run P (3 parallel)")
    subprocess.run(base + ["--run-name", RUN, "--phases", "P", "--mem-estimates-from", "newmodels_mem_estimates_v2",
                           "--large-gib", "99"], cwd=E, check=False)
    failed = [c["case_id"] for c in cases if status(RUN, c["case_id"]) in DEFER]
    other = [(c["case_id"], status(RUN, c["case_id"])) for c in cases if status(RUN, c["case_id"]) not in DEFER | {"ok"}]
    log(f"after main P: failed by memory/timeout {len(failed)} {failed}; other non-ok {other}")
    if failed:
        rd = E / "runs" / "newmodels_retry_est"
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "attempts.jsonl").write_text("".join(json.dumps({"case_id": c, "status": "ok", "mem_peak": 17 * GiB}) + "\n" for c in failed))
        log("retry: 2 parallel (P), then alone (S)")
        subprocess.run(base + ["--run-name", RUN + "_retry2", "--ids"] + failed +
                       ["--mem-estimates-from", "newmodels_retry_est", "--large-gib", "99", "--cpusets", "0-11,12-23",
                        "--par-threads", "12", "--par-mem", "40g", "--mem-budget-gib", "44"], cwd=E, check=False)
    log("NEWMODELS_CHAIN_DONE")


if __name__ == "__main__":
    main()
