"""Synthetic validation of the diagnostic replayer and of the pinned xMem allocator decisions.

Expected values come from an independent transcription of PyTorch v2.6.0
c10/cuda/CUDACachingAllocator.cpp (round_size / get_pool / get_allocation_size / should_split),
saved under provenance/pytorch_v2.6.0_ref. These are synthetic request sequences, not GPU data.
Run:  python tests/test_replayer_synthetic.py  (inside the pinned CPU image; see docs/COMMANDS.md)
"""
import io
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cgf_diag.replay import Replayer, time_grouped_xmem, MIB  # noqa: E402
from cgf_diag.events import normalize_gpu_snapshot, load_snapshot  # noqa: E402

KMIN, KSMALL, KSMALLBUF, KLARGEBUF, KMINLARGE, KROUND = 512, 1 << 20, 2 << 20, 20 << 20, 10 << 20, 2 << 20
RESULTS = {}


def pt26(size):
    rounded = KMIN if size < KMIN else KMIN * ((size + KMIN - 1) // KMIN)
    pool = "small" if rounded <= KSMALL else "large"
    if rounded <= KSMALL:
        seg = KSMALLBUF
    elif rounded < KMINLARGE:
        seg = KLARGEBUF
    else:
        seg = KROUND * ((rounded + KROUND - 1) // KROUND)
    return rounded, pool, seg


def rows_of(spec):
    """spec: list of (action, alloc_id, size[, t])"""
    out = []
    for i, item in enumerate(spec):
        act, aid, size = item[:3]
        t = item[3] if len(item) > 3 else float(i)
        out.append({"seq": i, "action": act, "alloc_id": aid, "size": size, "stream": 0, "t": t, "addr": 1000 + aid,
                    "phase": "p", "invocation": 0})
    return out


def first_decision(policy, size):
    r = Replayer(80 << 30, policy)
    r.run(rows_of([("alloc", 0, size)]))
    return r.decisions[0]


def test_size_boundaries():
    sizes = [1, 511, 512, 513, KSMALL - 1, KSMALL, KSMALL + 1, KMINLARGE - KMIN - 1, KMINLARGE - KMIN,
             KMINLARGE - 1, KMINLARGE, KMINLARGE + 1, 8519680, 1048576]
    rows = []
    for s in sizes:
        rounded, pool, seg = pt26(s)
        x = first_decision("xmem_pinned", s)
        d = first_decision("diag_pt26_alloc_size_boundary", s)
        rows.append({"request": s, "pt26_rounded": rounded, "pt26_pool": pool, "pt26_segment": seg,
                     "xmem_block": x[3], "xmem_segment": x[4], "diag_block": d[3], "diag_segment": d[4]})
        assert d[4] == seg, (s, d, seg)
        assert x[3] == d[3]
        # block given to the request = rounded size unless the segment remainder is too small to split
        exp_block = rounded if (seg - rounded >= KMIN if pool == "small" else seg - rounded > KSMALL) else seg
        assert d[3] == exp_block, (s, d[3], exp_block)
    mism = [r for r in rows if r["xmem_segment"] != r["pt26_segment"]]
    RESULTS["size_boundaries"] = {"rows": rows, "xmem_vs_pt26_segment_mismatch": mism}
    # Only (10MiB-511 .. 10MiB] round to exactly 10MiB: xMem 20MiB segment vs PyTorch 2.6 10MiB.
    assert all(KMINLARGE - KMIN < m["request"] <= KMINLARGE for m in mism), mism
    assert any(m["request"] == KMINLARGE and m["xmem_segment"] == KLARGEBUF and m["pt26_segment"] == KMINLARGE for m in mism)


def test_reuse_after_free():
    r = Replayer(80 << 30)
    r.run(rows_of([("alloc", 0, 4 * MIB), ("free", 0, 4 * MIB), ("alloc", 1, 4 * MIB)]))
    assert r.decisions[1][4] == 0 and r.reserved == KLARGEBUF


def test_split_and_merge():
    r = Replayer(80 << 30)
    r.run(rows_of([("alloc", 0, 4 * MIB), ("alloc", 1, 4 * MIB), ("free", 0, 4 * MIB), ("free", 1, 4 * MIB),
                   ("alloc", 2, 18 * MIB)]))
    assert [d[4] for d in r.decisions] == [KLARGEBUF, 0, 0], r.decisions  # merged block reused
    small = Replayer(80 << 30)
    small.run(rows_of([("alloc", 0, 600_000), ("alloc", 1, 600_000), ("free", 0, 600_000), ("free", 1, 600_000),
                       ("alloc", 2, KSMALL)]))
    # 1MiB fits only in the merged 2MiB small block (unmerged remainder would be 897,024 B)
    assert [d[4] for d in small.decisions] == [KSMALLBUF, 0, 0] and small.reserved == KSMALLBUF, small.decisions
    RESULTS["split_merge"] = {"large": [d[4] for d in r.decisions], "small_pool_reserved": small.reserved}


def test_capacity_release_and_retry():
    r = Replayer(40 * MIB)
    r.run(rows_of([("alloc", 0, 15 * MIB), ("free", 0, 15 * MIB), ("alloc", 1, 30 * MIB)]))
    assert r.oom is None
    assert r.decisions[1][4] == 30 * MIB and r.decisions[1][5] == 16 * MIB, r.decisions
    assert r.reserved == 30 * MIB
    r2 = Replayer(40 * MIB)
    r2.run(rows_of([("alloc", 0, 30 * MIB), ("alloc", 1, 20 * MIB)]))
    assert r2.oom is not None and r2.oom["size"] == 20 * MIB
    RESULTS["capacity"] = {"release_retry": r.decisions[1], "oom": r2.oom}


def test_same_timestamp_regrouping():
    spec = [("alloc", 0, 12 * MIB, 0.0), ("free", 0, 12 * MIB, 1.0), ("alloc", 1, 12 * MIB, 1.0), ("free", 1, 12 * MIB, 2.0)]
    seq = Replayer(80 << 30)
    seq.run(rows_of(spec))
    _, grouped_reserved = time_grouped_xmem(rows_of(spec), [], 80, first_region_ts=0.0)
    RESULTS["same_ts"] = {"sequential_peak_reserved": seq.peak_reserved, "time_grouped_peak_reserved": grouped_reserved}
    assert seq.peak_reserved == 12 * MIB and grouped_reserved == 24 * MIB


def test_streams_observation():
    """Observation (not a PyTorch equivalence claim): xMem pool lower_bound does not reject a block of a
    different stream (PyTorch get_free_block returns false when (*it)->stream != p.stream())."""
    r = Replayer(80 << 30)
    rows = rows_of([("alloc", 0, 4 * MIB), ("free", 0, 4 * MIB), ("alloc", 1, 4 * MIB)])
    rows[0]["stream"] = rows[1]["stream"] = 7  # first stream seen -> index 0
    rows[2]["stream"] = 9                      # second stream -> index 1
    r.run(rows)
    other_first = Replayer(80 << 30)
    rows2 = rows_of([("alloc", 0, 4 * MIB), ("free", 0, 4 * MIB), ("alloc", 1, 4 * MIB)])
    rows2[0]["stream"] = rows2[1]["stream"] = 9
    rows2[2]["stream"] = 7
    other_first._stream(7)  # index 0 assigned to 7 first
    other_first.run(rows2)
    RESULTS["streams"] = {"stream1_request_after_stream0_free_new_segment_bytes": r.decisions[1][4],
                          "stream0_request_after_stream1_free_new_segment_bytes": other_first.decisions[1][4],
                          "pytorch26_expected_new_segment_bytes": KLARGEBUF}


def test_gpu_importer_and_safe_pickle():
    snap = {"segments": [], "device_traces": [[
        {"action": "segment_alloc", "addr": 0, "size": KLARGEBUF, "stream": 0, "time_us": 1},
        {"action": "alloc", "addr": 0, "size": 4 * MIB, "stream": 0, "time_us": 1},
        {"action": "free_requested", "addr": 0, "size": 4 * MIB, "stream": 0, "time_us": 2},
        {"action": "free_completed", "addr": 0, "size": 4 * MIB, "stream": 0, "time_us": 2},
        {"action": "alloc", "addr": 0, "size": 3 * MIB, "stream": 0, "time_us": 3},
        {"action": "free_requested", "addr": 0, "size": 3 * MIB, "stream": 0, "time_us": 4},
        {"action": "free_completed", "addr": 0, "size": 3 * MIB, "stream": 0, "time_us": 4},
    ]]}
    bounds = [{"phase": "load", "invocation": 0, "counters": {"alloc": 0, "free": 0, "segment_alloc": 0, "segment_free": 0}},
              {"phase": "gen", "invocation": 0, "counters": {"alloc": 1, "free": 1, "segment_alloc": 1, "segment_free": 0}}]
    rows, stats, _ = normalize_gpu_snapshot(snap, boundaries=bounds)
    assert [r["alloc_id"] for r in rows if r["action"] == "alloc"] == [0, 1]
    assert rows[4]["phase"] == "gen" and rows[1]["phase"] == "load"
    assert stats["unmatched_frees"] == 0 and stats["live_at_end"] == 0 and not stats["boundaries_unplaced"]
    r = Replayer(80 << 30)
    r.run(rows)
    assert [d[4] for d in r.decisions] == [KLARGEBUF, 0]

    class Evil:
        def __reduce__(self):
            return (print, ("pwned",))
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pickle") as f:
        f.write(pickle.dumps({"x": Evil()}))
        f.flush()
        try:
            load_snapshot(f.name)
            raise AssertionError("unsafe pickle accepted")
        except pickle.UnpicklingError:
            pass


def test_gpu_replay_pipeline_first_divergence():
    """Synthetic run dir shaped like the collector output. The fixture follows PyTorch 2.6 rules: an
    exact 10MiB request gets a 10MiB segment. xMem (20MiB) must diverge at request 0; the diagnostic
    boundary policy must not. An explicit empty_cache marker must be replayed as an API call."""
    import subprocess, tempfile, gzip as _gz
    tmp = Path(tempfile.mkdtemp())
    run = tmp / "run"
    run.mkdir()
    tr = [
        {"action": "segment_alloc", "addr": 0, "size": 10 * MIB, "stream": 0, "time_us": 1},
        {"action": "alloc", "addr": 0, "size": 10 * MIB, "stream": 0, "time_us": 1},
        {"action": "free_requested", "addr": 0, "size": 10 * MIB, "stream": 0, "time_us": 2},
        {"action": "free_completed", "addr": 0, "size": 10 * MIB, "stream": 0, "time_us": 2},
        {"action": "segment_free", "addr": 0, "size": 10 * MIB, "stream": 0, "time_us": 3},   # empty_cache
        {"action": "segment_alloc", "addr": 0, "size": 20 * MIB, "stream": 0, "time_us": 4},
        {"action": "alloc", "addr": 0, "size": 3 * MIB, "stream": 0, "time_us": 4},
        {"action": "free_requested", "addr": 0, "size": 3 * MIB, "stream": 0, "time_us": 5},
        {"action": "free_completed", "addr": 0, "size": 3 * MIB, "stream": 0, "time_us": 5},
    ]
    (run / "memory_snapshot.pickle").write_bytes(pickle.dumps({"segments": [], "device_traces": [tr]}))
    z = {"alloc": 0, "free": 0, "segment_alloc": 0, "segment_free": 0}
    bounds = [
        {"kind": "record_start", "counters": z, "memory_stats": {"reserved_bytes.all.current": 0}},
        {"kind": "phase_begin", "phase": "model_load", "invocation": 0, "counters": z},
        {"kind": "empty_cache_before", "phase": "model_load", "invocation": 0,
         "counters": {"alloc": 1, "free": 1, "segment_alloc": 1, "segment_free": 0}},
        {"kind": "phase_begin", "phase": "single/measure", "invocation": 0,
         "counters": {"alloc": 1, "free": 1, "segment_alloc": 1, "segment_free": 1}},
        {"kind": "record_end", "counters": {"alloc": 2, "free": 2, "segment_alloc": 2, "segment_free": 1},
         "memory_stats": {}},
    ]
    (run / "boundaries.jsonl").write_text("\n".join(json.dumps(b) for b in bounds))
    (run / "run_meta.json").write_text(json.dumps({"run_id": "synthetic", "case_id": "synthetic"}))
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-m", "cgf_diag.gpu_replay", "--run-dir", str(run), "--out", str(tmp / "out"),
                    "--xmem-lock", str(root / "provenance/xmem-allocator-lock.json")], check=True,
                   env={**__import__("os").environ, "PYTHONPATH": f"{root / 'src'}:" + __import__("os").environ.get("PYTHONPATH", "")})
    rep = json.loads((tmp / "out/import_report.json").read_text())
    x = rep["policies"]["xmem_pinned"]["comparison"]
    d = rep["policies"]["diag_pt26_alloc_size_boundary"]["comparison"]
    RESULTS["gpu_pipeline"] = {"import": {k: rep["import"][k] for k in ("initial_state", "trace_complete", "empty_cache_markers")},
                               "xmem_first_divergence": x["first_decision_divergence_request_index"],
                               "diag_first_divergence": d["first_decision_divergence_request_index"],
                               "diag_mismatches": d["decision_mismatches"]}
    assert rep["import"]["initial_state"] == "empty" and rep["import"]["trace_complete"] and rep["import"]["empty_cache_markers"] == 1
    assert x["first_decision_divergence_request_index"] == 0
    assert d["decision_mismatches"] == 0, d


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS", t.__name__)
        except Exception as exc:  # report all
            failed += 1
            print("FAIL", t.__name__, type(exc).__name__, exc)
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out:
        out.write_text(json.dumps({"failed": failed, "results": RESULTS}, indent=1, default=str))
    sys.exit(1 if failed else 0)
