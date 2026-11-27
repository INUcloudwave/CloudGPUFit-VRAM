#!/usr/bin/env python3
"""Reserved prediction v3 over stored CPU request streams (post-processing; CPU traces and result.json untouched).

Cases: every case of the L256 table (runs/main395_l256/cases) and the L1024-only cases (runs/main395/cases) whose
CPU worker finished ok. Per case:
  main   : policy descending + cuBLAS workspace at the probed position (cgf680.gemm_probe --exact, CPU only)
           -> R_inf, R_at{effective, cuda_total}, R_min
  ablate : (a) stock xMem addresses, no workspace (= v2 replay) ; (b) descending, no workspace
           -> R_inf and R_at effective only
Run inside the pinned image, e.g.
  MEM=40g CPUS=20 docs/run_in_image.sh tools/reserved_predict.py --name rtx3090_v3 --gpu rtx3090 --jobs 20
Output: results/reserved_v3/<name>/predictions.jsonl (resumable), run_info.json
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

EXP = Path("/exp")
ABLATIONS = {"stock_nows": ("xmem", False), "desc_nows": ("descending", False)}


def case_list(runs_root=None, cases_rel=None):
    if runs_root:                                   # single run (e.g. AWS fp16_real): one table, one case root
        cs = [json.loads(x) for x in (EXP / cases_rel).read_text().splitlines()]
        return [(c, EXP / runs_root / "cases" / c["case_id"], Path(cases_rel).stem) for c in cs]
    l256 = [json.loads(x) for x in (EXP / "manifest/cases395_l256.jsonl").read_text().splitlines()]
    l1024 = [json.loads(x) for x in (EXP / "manifest/cases395.jsonl").read_text().splitlines()]
    ids256 = {c["case_id"] for c in l256}
    ids1024 = {c["case_id"] for c in l1024}
    out = []
    for c in l256:
        out.append((c, EXP / "runs/main395_l256/cases" / c["case_id"],
                    "both" if c["case_id"] in ids1024 else "L256"))
    for c in l1024:
        if c["case_id"] not in ids256:
            out.append((c, EXP / "runs/main395/cases" / c["case_id"], "L1024"))
    return out


def work(arg):
    case, cdir, table, ws_rel, gpus, want_ablate = arg
    from cgf680 import replay3
    first = next(iter(gpus))
    caps, ws_b = gpus[first]["caps"], gpus[first]["ws_bytes"]
    t0 = time.time()
    row = {"case_id": case["case_id"], "model": case["short"], "kind": case["kind"], "table": table,
           "case_dir": str(cdir.relative_to(EXP)), "ws_rel": ws_rel}
    try:
        res = json.loads((cdir / "result.json").read_text())
        row["v2"] = {k: res["summary"][k] for k in ("first_measure_start_reserved", "first_measure_peak_reserved",
                                                    "first_measure_peak_allocated", "after_load_reserved")}
        st = replay3.load_stream(cdir)
        row["main"] = replay3.predict(cdir, caps, "descending", ws_rel, want_rmin=True, ws_bytes=ws_b, st=st)
        row["main"]["gpu"] = first
        if len(gpus) > 1:                    # other target GPUs: own capacity and cuBLAS workspace size
            row["by_gpu"] = {first: row["main"]}
            for g, gi in list(gpus.items())[1:]:
                row["by_gpu"][g] = replay3.predict(cdir, gi["caps"], "descending", ws_rel, want_rmin=True,
                                                   ws_bytes=gi["ws_bytes"], st=st)
        if want_ablate and "effective" in caps:
            row["ablate"] = {}
            for name, (pol, ws) in ABLATIONS.items():
                p = replay3.predict(cdir, {"effective": caps["effective"]}, pol, ws_rel if ws else None, want_rmin=False,
                                    ws_bytes=ws_b, st=st)
                row["ablate"][name] = {"R_inf": p["R_inf"], "R_at_effective": p["R_at"]["effective"]}
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"[:500]
        row["traceback"] = traceback.format_exc()[-2000:]
    row["wall_s"] = round(time.time() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--gpu", default="rtx3090", help="profile name(s) in manifest/gpu_profiles.json, comma separated; the first is 'main'")
    ap.add_argument("--ws-positions", default="runs/gemm_exact/positions.jsonl",
                    help="cgf680.gemm_probe --exact output (per case; last row wins; status ok required)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default=None, help="comma separated case ids")
    ap.add_argument("--no-ablate", action="store_true")
    ap.add_argument("--runs-root", default=None, help="e.g. runs/aws_fp16_l256 (default: INUOS main395_l256 + main395)")
    ap.add_argument("--cases", default="manifest/cases395_l256.jsonl", help="table for --runs-root")
    ap.add_argument("--ws-source", choices=("positions", "result"), default="positions",
                    help="result: cublas_workspace_position.ws_insert_rel written by cpu_worker2 --dtype-mode fp16_real")
    a = ap.parse_args()
    from cgf_diag.replay import verify_xmem
    xm = verify_xmem("/work/xMem", "/exp/provenance/xmem-allocator-lock.json")
    from cgf680.gpu_rules import cublas_workspace
    profiles = json.loads((EXP / "manifest/gpu_profiles.json").read_text())["profiles"]
    gpus = {}
    for g in a.gpu.split(","):
        prof = profiles[g]
        caps = {}
        if prof.get("capacity_bytes"):
            caps["effective"] = prof["capacity_bytes"]
        if prof.get("cuda_total_memory"):
            caps["cuda_total"] = prof["cuda_total_memory"]
        ws = cublas_workspace(prof["gpu_name"], prof.get("compute_capability"), prof.get("runtime_torch", "2.6.0"))
        gpus[g] = {"caps": caps, "ws_bytes": ws["bytes"], "ws_rule": ws}
    caps = gpus[next(iter(gpus))]["caps"]
    wsp = {}
    if a.ws_source == "positions":
        for line in (EXP / a.ws_positions).read_text().splitlines():
            r = json.loads(line)
            wsp[r["case_id"]] = r
    out_dir = EXP / "results/reserved_v3" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "predictions.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") == "ok":
                done.add(r["case_id"])
    only = set(a.only.split(",")) if a.only else None
    todo, skipped = [], []
    for case, cdir, table in case_list(a.runs_root, a.cases if a.runs_root else None):
        if only and case["case_id"] not in only:
            continue
        if case["case_id"] in done:
            continue
        rj = cdir / "result.json"
        if not (cdir / "invocation_fp16_requests.npz").exists() or not rj.exists() or \
                json.loads(rj.read_text()).get("status") != "ok":
            skipped.append(case["case_id"])
            continue
        if a.ws_source == "result":
            wp = json.loads(rj.read_text()).get("cublas_workspace_position") or {}
            pr = {"status": "ok", "prefix_equal_to_npz": True, "ws_insert_rel": wp.get("ws_insert_rel")} \
                if wp.get("ws_insert_rel") is not None else None
        else:
            pr = wsp.get(case["case_id"])
        if not pr or pr.get("status") != "ok" or not pr.get("prefix_equal_to_npz"):
            skipped.append(case["case_id"] + ":no_ws_position")
            continue
        ws_rel = pr["ws_insert_rel"]
        size = (cdir / "invocation_fp16_requests.npz").stat().st_size
        todo.append((size, (case, cdir, table, ws_rel, {g: {"caps": v["caps"], "ws_bytes": v["ws_bytes"]} for g, v in gpus.items()},
                            not a.no_ablate)))
    todo.sort(key=lambda x: -x[0])                     # large streams first
    info = {"name": a.name, "gpu": a.gpu, "caps": caps, "gpus": gpus, "xmem": xm, "ws_positions": a.ws_positions, "ws_source": a.ws_source,
            "runs_root": a.runs_root,
            "ws_positions_n": len(wsp), "jobs": a.jobs, "todo": len(todo), "already_done": len(done),
            "skipped_not_ok": skipped, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "argv": sys.argv}
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=1))
    print(f"todo {len(todo)} done {len(done)} skipped {len(skipped)}", flush=True)
    with mp.get_context("fork").Pool(a.jobs, maxtasksperchild=1) as pool, out.open("a") as f:
        for i, row in enumerate(pool.imap_unordered(work, [t[1] for t in todo]), 1):
            f.write(json.dumps(row) + "\n")
            f.flush()
            m = row.get("main", {})
            print(f"[{i}/{len(todo)}] {row['case_id']} {row['status']} R_inf={m.get('R_inf')} "
                  f"R_eff={m.get('R_at', {}).get('effective', {}).get('reserved')} R_min={m.get('R_min')} {row['wall_s']}s",
                  flush=True)
    print("RESERVED_PREDICT_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
