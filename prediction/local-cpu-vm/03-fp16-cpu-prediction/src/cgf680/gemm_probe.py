"""GEMM probe (reserved prediction v3): where the GPU-only cuBLAS workspace request goes in a CPU request stream.

GPU traces (INUGPU 2026-11-16) show the cuBLAS workspace (8,519,680 B, never freed) allocated right after the
output of the FIRST cuBLAS call of warmup_0 (the first matmul-type torch function: linear, matmul, addmm, bmm ...).
This probe runs the same CPU pipeline as cpu_worker2 (FP32 execution, USER_SCOPE profiler, shadow FP16 sizes) on a
small variant of a case and reports, inside the FP16 invocation request stream, the index right after the output
request of the first GEMM-type call.  The index is relative to the invocation start (marker M0), so it can be applied
to the stored invocation_fp16_requests.npz of any case of the same model.  Nothing of the production worker changes.

  python -m cgf680.gemm_probe --case-id <case> --batch 1 --input-len 128 --new-tokens 2 --out <dir>
"""
from __future__ import annotations

import argparse
import bisect
import gc
import json
import sys
import time
from pathlib import Path

import torch

from cgf680 import protocol2 as P
from cgf680.cpu_worker import MARK_BASE, MARKERS, Session, normalize, parse_trace
from cgf680.cpu_worker2 import REGION
from cgf680.shadow import emulate_fp16, match_records, patch_fp16_branches

from cgf680.gemm_shadow import GEMM_NAMES, FirstGemmReached, NamedShadow  # noqa: E402


def record_hits(ev, records):
    """Same matching rule as shadow.match_records; returns {event index: call number} for matched records."""
    act, addr, size = ev["action"], ev["addr"], ev["size"]
    by_addr = {}
    for i in range(len(act)):
        if act[i] == 1:
            by_addr.setdefault(int(addr[i]), []).append(i)
    used, hits = set(), {}
    floor, cur_call, call_max = -1, None, -1
    for kind, p, nbytes, x, call in records:
        if call != cur_call:
            floor = max(floor, call_max)
            cur_call = call
        lst = by_addr.get(int(p), [])
        j = bisect.bisect_right(lst, floor)
        while j < len(lst):
            k = lst[j]
            if k not in used and int(size[k]) == int(nbytes):
                used.add(k)
                call_max = max(call_max, k)
                hits[k] = (kind, call)
                break
            j += 1
    return hits


def probe_exact(case_ids, cases_path, models_path, npz_root, threads, out_path, abort_after=8):
    """Exact mode: run each case with its own B / lengths / new tokens in a fresh process state, stop right after the
    first GEMM call, and check that the probed request prefix equals the stored stream prefix (npz)."""
    import numpy as np
    cases = {c["case_id"]: c for c in map(json.loads, Path(cases_path).read_text().splitlines())}
    models = json.loads(Path(models_path).read_text())["models"]
    torch.set_num_threads(threads)
    rows = []
    model_cache = {}
    for cid in case_ids:
        case = cases[cid]
        t0 = time.time()
        row = {"case_id": cid, "model": case["short"]}
        try:
            torch.manual_seed(P.SEED)
            if case["model"] not in model_cache:
                model_cache.clear()
                gc.collect()
                m16 = P.load(case, models[case["model"]], torch.float16)   # cpu_worker2 step 1 (structure for load synthesis)
                del m16
                gc.collect()
                m = P.finish_load(P.load(case, models[case["model"]], torch.float32), torch.float32)
                row["fp16_branch_patches"] = patch_fp16_branches(m)
                model_cache[case["model"]] = m
                row["first_case_of_process_model"] = True
            m = model_cache[case["model"]]
            bad = P.check_valid(case, m.config)
            if bad:
                raise P.InvalidInput(bad)
            host, _ = P.host_inputs(case, m.config)
            inputs = {k: (v.to(torch.float32) if v.is_floating_point() else v) for k, v in host.items()}
            if case["kind"] != "vision_cls":
                P.generation_config(m, case)      # same as cpu_worker2 (sets model.generation_config)
            shadow = NamedShadow(abort=abort_after > 0, abort_after=abort_after)

            def mark(k):
                x = torch.empty(MARK_BASE + 2 * k, dtype=torch.uint8)
                del x
            gc.collect()
            gc.disable()
            sess = Session("user")
            sess.start()
            reached = None
            try:
                mark(0)
                try:
                    with torch.profiler.record_function(REGION):
                        with shadow:
                            with torch.inference_mode():
                                P.invoke(m, case, inputs)
                except FirstGemmReached as e:
                    reached = str(e)
            finally:
                sess.stop()
                gc.enable()
            raw = Path(out_path).with_suffix(f".{cid}.trace.json")
            sess.export(str(raw))
            del sess
            regions, mem = parse_trace(raw)
            raw.unlink()
            ev, mpos, _ = normalize(regions, mem, len(MARKERS))
            tags, buf_src, _ = match_records(ev, shadow.records)
            ev16, _, idx = emulate_fp16(ev, tags, buf_src)
            hits = record_hits(ev, shadow.records)
            p0 = idx[mpos[0]]
            first = next((i for i in sorted(hits) if hits[i][0] == "out" and shadow.names.get(hits[i][1]) in GEMM_NAMES), None)
            if first is None:
                row["status"] = "no_gemm_found"
            else:
                rel = idx[first] - p0
                z = np.load(Path(npz_root) / cid / "invocation_fp16_requests.npz")
                pa, ps = ev16["action"][p0:p0 + rel + 1], ev16["size"][p0:p0 + rel + 1]
                same = bool(len(z["action"]) > rel and np.array_equal(pa, z["action"][:rel + 1]) and
                            np.array_equal(ps, z["size"][:rel + 1]))
                row.update(status="ok" if same else "prefix_mismatch", gemm_func=shadow.names[hits[first][1]],
                           ws_insert_rel=int(rel) + 1,
                           first_gemm_output_bytes_fp16=int(ev16["size"][idx[first]]), prefix_equal_to_npz=same)
                if not same:
                    k = next((i for i in range(min(len(pa), len(z["action"])))
                              if pa[i] != z["action"][i] or ps[i] != z["size"][i]), None)
                    row["first_prefix_diff"] = k
                    row["probe_prefix"] = [[int(x), int(y)] for x, y in zip(pa[:120], ps[:120])]
                    row["npz_prefix"] = [[int(x), int(y)] for x, y in zip(z["action"][:120], z["size"][:120])]
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"[:400]
        row["wall_s"] = round(time.time() - t0, 1)
        rows.append(row)
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print("CGF_GEMM_EXACT " + json.dumps({k: row.get(k) for k in ("case_id", "status", "gemm_func", "ws_insert_rel",
                                                                     "prefix_equal_to_npz", "wall_s")}), flush=True)
    return rows


def main():
    if "--exact" in sys.argv:
        ap = argparse.ArgumentParser()
        ap.add_argument("--exact", action="store_true")
        ap.add_argument("--case-ids", required=True, help="comma separated, or @file with one id per line")
        ap.add_argument("--cases", default="/exp/manifest/cases395_l256.jsonl")
        ap.add_argument("--models", default="/exp/manifest/models.json")
        ap.add_argument("--npz-root", required=True)
        ap.add_argument("--threads", type=int, default=4)
        ap.add_argument("--abort-after", type=int, default=8, help="stop after N GEMM calls (0: run the whole invocation)")
        ap.add_argument("--out", required=True, help="jsonl (appended)")
        a = ap.parse_args()
        ids = (Path(a.case_ids[1:]).read_text().split() if a.case_ids.startswith("@") else a.case_ids.split(","))
        probe_exact(ids, a.cases, a.models, a.npz_root, a.threads, a.out, a.abort_after)
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--cases", default="/exp/manifest/cases395_l256.jsonl")
    ap.add_argument("--models", default="/exp/manifest/models.json")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--input-len", type=int, default=None)
    ap.add_argument("--new-tokens", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    case = dict(next(c for c in map(json.loads, Path(a.cases).read_text().splitlines()) if c["case_id"] == a.case_id))
    case["batch_size"] = a.batch
    if case["kind"] != "vision_cls":
        case["input_len"] = a.input_len or case["input_len"]
        case["new_tokens"] = a.new_tokens
    pin = json.loads(Path(a.models).read_text())["models"][case["model"]]
    torch.set_num_threads(a.threads)
    torch.manual_seed(P.SEED)
    R = {"source_case": a.case_id, "probe_case": case, "gemm_names": sorted(GEMM_NAMES), "status": "running"}
    t0 = time.time()
    m = P.finish_load(P.load(case, pin, torch.float32), torch.float32)
    host, _ = P.host_inputs(case, m.config)
    inputs = {k: (v.to(torch.float32) if v.is_floating_point() else v) for k, v in host.items()}
    if case["kind"] != "vision_cls":
        P.generation_config(m, case)
    R["fp16_branch_patches"] = patch_fp16_branches(m)
    shadow = NamedShadow()

    def mark(k):
        x = torch.empty(MARK_BASE + 2 * k, dtype=torch.uint8)
        del x
    gc.collect()
    gc.disable()
    sess = Session("user")
    sess.start()
    try:
        mark(0)
        with torch.profiler.record_function(REGION):
            with shadow:
                with torch.inference_mode():
                    gi, o, new = P.invoke(m, case, inputs)
        mark(4)
        del gi, o
        mark(5)
    finally:
        sess.stop()
        gc.enable()
    raw = out / "probe_trace.json"
    sess.export(str(raw))
    del sess
    regions, mem = parse_trace(raw)
    ev, mpos, _ = normalize(regions, mem, len(MARKERS))
    raw.unlink()
    tags, buf_src, _ = match_records(ev, shadow.records)
    ev16, _, idx = emulate_fp16(ev, tags, buf_src)
    hits = record_hits(ev, shadow.records)
    p0 = idx[mpos[0]]
    first = None
    for i in sorted(hits):
        kind, call = hits[i]
        if kind == "out" and shadow.names.get(call) in GEMM_NAMES:
            first = i
            break
    if first is None:
        R["status"] = "no_gemm_found"
    else:
        rel = idx[first] - p0
        a16 = ev16["action"]
        R.update(status="ok", gemm_func=shadow.names[hits[first][1]], gemm_call=hits[first][1],
                 first_gemm_output_request_rel=int(rel), first_gemm_output_bytes_fp16=int(ev16["size"][idx[first]]),
                 ws_insert_rel=int(rel) + 1,
                 allocs_before_gemm_output=int((a16[p0:idx[first]] == 1).sum()),
                 names_of_first_calls=[shadow.names.get(c) for c in range(1, min(40, shadow.calls) + 1)])
    R["wall_s"] = round(time.time() - t0, 1)
    (out / "gemm_probe.json").write_text(json.dumps(R, indent=1, default=str))
    print("CGF_GEMM_PROBE " + json.dumps({k: R.get(k) for k in ("source_case", "status", "gemm_func", "ws_insert_rel",
                                                                 "first_gemm_output_bytes_fp16")}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
