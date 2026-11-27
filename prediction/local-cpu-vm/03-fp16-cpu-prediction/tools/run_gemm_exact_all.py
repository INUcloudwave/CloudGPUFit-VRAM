#!/usr/bin/env python3
"""Run the exact GEMM probe (cgf680.gemm_probe --exact) for every ok CPU case of the L256 table and the L1024-only
cases. Cases are grouped by (table, model, torch threads of the original CPU run) — CPU kernels allocate per-thread
scratch, so the probe must use the same thread count to reproduce the stored request prefix. One container per group,
fresh process; a case whose prefix differs from its stored npz is retried alone in a fresh process.
Host side (no numpy needed). Output: runs/gemm_exact/positions.jsonl (+ console log)
  python3 tools/run_gemm_exact_all.py --cpu-budget 22
"""
import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
TABLES = [("L256", "manifest/cases395_l256.jsonl", "runs/main395_l256/cases"),
          ("L1024", "manifest/cases395.jsonl", "runs/main395/cases")]


def groups():
    seen = set()
    g = defaultdict(list)
    for table, man, root in TABLES:
        for c in map(json.loads, (EXP / man).read_text().splitlines()):
            if c["case_id"] in seen:
                continue
            seen.add(c["case_id"])
            d = EXP / root / c["case_id"]
            try:
                r = json.loads((d / "result.json").read_text())
            except Exception:
                continue
            if r.get("status") != "ok" or not (d / "invocation_fp16_requests.npz").exists():
                continue
            g[(table, man, root, c["short"], int(r["environment"]["threads"]))].append(c["case_id"])
    return g


def run(key, ids, out, tag):
    table, man, root, model, threads = key
    lst = EXP / "runs/gemm_exact/lists" / f"{tag}.txt"
    lst.parent.mkdir(parents=True, exist_ok=True)
    lst.write_text("\n".join(ids) + "\n")
    env = {"CPUS": str(threads), "MEM": "48g" if model in ("t5-3b", "gpt2-xl") else "24g", "ABORT_AFTER": "8"}
    t0 = time.time()
    p = subprocess.run(["bash", str(EXP / "docs/run_gemm_exact.sh"), man, root, out, "@" + str(lst.relative_to(EXP))],
                       env={**__import__("os").environ, **env}, capture_output=True, text=True)
    lines = [x for x in p.stdout.splitlines() if x.startswith("CGF_GEMM_EXACT")]
    print(f"{tag}: {len(lines)}/{len(ids)} rc={p.returncode} {time.time() - t0:.0f}s", flush=True)
    if p.returncode != 0:
        print(p.stderr[-1500:], flush=True)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-budget", type=int, default=22)
    ap.add_argument("--mem-budget-gib", type=int, default=36)
    ap.add_argument("--out", default="runs/gemm_exact/positions.jsonl")
    a = ap.parse_args()
    g = groups()
    print(f"groups {len(g)} cases {sum(len(v) for v in g.values())}", flush=True)
    jobs = sorted(g.items(), key=lambda kv: -kv[0][4])
    # simple CPU-budget scheduler: run groups concurrently while the sum of threads fits the budget
    import threading
    lock, used, mused = threading.Lock(), [0], [0]
    cond = threading.Condition(lock)
    mem_est = {"t5-3b": 26, "gpt2-xl": 16, "qwen2.5-1.5b-instruct": 10, "tinyllama-1.1b-chat": 8, "vit-large-patch16-224": 6}

    def submit(key, ids, tag):
        need = min(key[4], a.cpu_budget)
        mneed = min(mem_est.get(key[3], 5), a.mem_budget_gib)
        with cond:
            while used[0] + need > a.cpu_budget or mused[0] + mneed > a.mem_budget_gib:
                cond.wait()
            used[0] += need
            mused[0] += mneed
        try:
            return run(key, ids, a.out, tag)
        finally:
            with cond:
                used[0] -= need
                mused[0] -= mneed
                cond.notify_all()

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(submit, k, v, f"{k[0]}__{k[3]}__t{k[4]}") for k, v in jobs]
        for f in futs:
            f.result()
    # retry prefix mismatches / errors alone in fresh processes
    rows = {}
    for line in (EXP / a.out).read_text().splitlines():
        r = json.loads(line)
        rows[r["case_id"]] = r
    bad = [cid for cid, r in rows.items() if r["status"] != "ok"]
    print(f"first pass: ok {sum(r['status'] == 'ok' for r in rows.values())} not ok {len(bad)}", flush=True)
    key_of = {cid: k for k, v in g.items() for cid in v}
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(lambda cid: submit(key_of[cid], [cid], f"retry__{cid}"), bad))
    rows = {}
    for line in (EXP / a.out).read_text().splitlines():
        r = json.loads(line)
        rows[r["case_id"]] = r          # last row wins
    print(f"final: ok {sum(r['status'] == 'ok' for r in rows.values())} / {len(rows)}; "
          f"not ok {[c for c, r in rows.items() if r['status'] != 'ok']}", flush=True)
    print("GEMM_EXACT_ALL_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
