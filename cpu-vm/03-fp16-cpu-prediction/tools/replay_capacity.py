#!/usr/bin/env python3
"""Replay saved CPU request traces with a GPU allocator capacity (post-processing; the CPU trace and result.json
are not modified). Same order as cpu_worker2.replay_protocol: load-phase events -> traced invocation x4
(warmup 0,1,2 + measure 0), allocator cache kept, xMem pinned CachingAllocator (cgf_diag.replay.Replayer).
At capacity xMem frees cached segments and retries (release_cached_blocks), then raises -> OOM prediction.

Run inside the pinned image (docs/run_in_image.sh), e.g.
  docs/run_in_image.sh tools/replay_capacity.py --run main395_l256 --gpu rtx3090 --jobs 4
  docs/run_in_image.sh tools/replay_capacity.py --run main395_l256 --cap-bytes 1099511627776 --self-check
Output: results/<run>/capacity_<gpu>.jsonl (one row per ok case)
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

EXP = Path("/exp")
N_WARMUP = 3


def replay(cdir, cap):
    from cgf_diag.replay import Replayer
    z = np.load(cdir / "invocation_fp16_requests.npz")
    act, size, aid, m4 = z["action"], z["size"], z["alloc_id"], int(z["m4_local"])
    rp = Replayer(cap, policy="xmem_pinned", keep_decisions=False)
    i, releases = 0, []
    for a, s, x in z["load_events"].tolist():
        rp.apply({"seq": i, "action": "alloc" if a else "free", "size": s, "stream": 0, "alloc_id": ("L", x),
                  "phase": "load", "invocation": None})
        i += 1
    after_load = {"reserved": rp.reserved, "allocated": rp.allocated}
    runs = []
    for k in range(N_WARMUP + 1):
        name = "measure_0" if k == N_WARMUP else f"warmup_{k}"
        st = {"reserved": rp.reserved, "allocated": rp.allocated}
        pr, pa, end, n_rel, rel_bytes = st["reserved"], st["allocated"], None, 0, 0
        for j in range(len(act)):
            if j == m4:
                end = {"reserved": rp.reserved, "allocated": rp.allocated}
            before = rp.reserved
            ok = rp.apply({"seq": i, "action": "alloc" if act[j] else "free", "size": int(size[j]), "stream": 0,
                           "alloc_id": (k, int(aid[j])), "phase": name, "invocation": k})
            i += 1
            if not ok:
                break
            r, al = rp.reserved, rp.allocated
            if act[j] and r < before:      # reserved fell during a malloc -> cached segments were released
                n_rel += 1
                rel_bytes += before - r
            if r > pr:
                pr = r
            if al > pa:
                pa = al
        if end is None:
            end = {"reserved": rp.reserved, "allocated": rp.allocated}
        runs.append({"run": name, "start": st, "peak": {"reserved": pr, "allocated": pa}, "end_with_output": end,
                     "released": {"reserved": rp.reserved, "allocated": rp.allocated},
                     "cache_release_events": n_rel, "cache_released_bytes": rel_bytes})
        if rp.oom:
            break
    return after_load, runs, rp.oom


def one(args):
    cid, cap, label = args
    cdir = EXP / "runs" / RUN / "cases" / cid
    t = time.time()
    try:
        r = json.loads((cdir / "result.json").read_text())
        al, runs, oom = replay(cdir, cap)
        m0 = runs[-1] if runs[-1]["run"] == "measure_0" and not oom else None
        row = {"condition_id": cid, "gpu": label, "capacity_bytes": cap, "oom_predicted": bool(oom),
               "oom_stage": runs[-1]["run"] if oom else None, "oom_detail": oom,
               "after_load_reserved": al["reserved"], "after_load_allocated": al["allocated"],
               "first_measure_start_reserved": m0["start"]["reserved"] if m0 else None,
               "first_measure_start_allocated": m0["start"]["allocated"] if m0 else None,
               "first_measure_peak_reserved": m0["peak"]["reserved"] if m0 else None,
               "first_measure_peak_allocated": m0["peak"]["allocated"] if m0 else None,
               "cache_release_events": sum(x["cache_release_events"] for x in runs),
               "cache_release_by_run": {x["run"]: x["cache_release_events"] for x in runs},
               "worker_summary_1TiB_peak_reserved": (r.get("summary") or {}).get("first_measure_peak_reserved"),
               "seconds": round(time.time() - t, 1)}
        if SELF_CHECK:
            s = r.get("summary") or {}
            row["self_check_equal"] = all(row[k] == s.get(k) for k in (
                "first_measure_start_reserved", "first_measure_start_allocated", "first_measure_peak_reserved",
                "first_measure_peak_allocated", "after_load_reserved", "after_load_allocated"))
        return row
    except Exception as exc:  # noqa: BLE001
        return {"condition_id": cid, "gpu": label, "error": repr(exc)[:500]}


def init(run, self_check):
    global RUN, SELF_CHECK
    RUN, SELF_CHECK = run, self_check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main395_l256")
    ap.add_argument("--gpu", default=None, help="profile name in manifest/gpu_profiles.json")
    ap.add_argument("--cap-bytes", type=int, default=None, help="explicit capacity (label cap<bytes>)")
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--self-check", action="store_true", help="compare with result.json summary (use 1 TiB)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.gpu:
        cap, label = json.loads((EXP / "manifest/gpu_profiles.json").read_text())["profiles"][a.gpu]["capacity_bytes"], a.gpu
    elif a.cap_bytes:
        cap, label = a.cap_bytes, f"cap{a.cap_bytes}"
    else:
        sys.exit("--gpu or --cap-bytes required")
    root = EXP / "runs" / a.run / "cases"
    ids = []
    for d in sorted(root.iterdir()):
        if ".attempt-" in d.name or (a.ids is not None and d.name not in a.ids):
            continue
        rp = d / "result.json"
        if rp.exists() and json.loads(rp.read_text()).get("status") == "ok" and (d / "invocation_fp16_requests.npz").exists():
            ids.append(d.name)
    out = Path(a.out) if a.out else EXP / "results" / a.run / f"capacity_{label}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["condition_id"] for l in out.read_text().splitlines() if "error" not in json.loads(l)}
    todo = [c for c in ids if c not in done]
    print(json.dumps({"ok_cases": len(ids), "already": len(done), "todo": len(todo), "capacity_bytes": cap, "out": str(out)}), flush=True)
    with mp.Pool(a.jobs, initializer=init, initargs=(a.run, a.self_check)) as pool, out.open("a") as f:
        for row in pool.imap_unordered(one, [(c, cap, label) for c in todo]):
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(json.dumps({k: row.get(k) for k in ("condition_id", "oom_predicted", "first_measure_peak_reserved",
                                                       "cache_release_events", "self_check_equal", "error", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
