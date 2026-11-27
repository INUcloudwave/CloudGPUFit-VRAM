#!/usr/bin/env python3
"""Verify INUGPU reserved-guide claims (read-only analysis, nothing in the experiment tree is modified).

For each condition with a GPU trace:
  A. GPU request stream -> xMem replay, policy stock / descending  (reproduce guide §3.1, §3.3)
  B. CPU request stream (load_events + 4x invocation from invocation_fp16_requests.npz) -> same replays
  C. request-list diff CPU vs GPU (load window, warmup_0 window)
usage: python verify_guide.py <out.jsonl> <cid>=<gpu_trace_dir>=<cpu_case_dir> ...
"""
import gzip, json, logging, pickle, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

logging.disable(logging.CRITICAL)
from perf_estimator.allocator.allocator import CachingAllocator, Device, Stream, Trace  # noqa: E402


class LightTrace(Trace):
    """same counters (max_segment_changes / max_usage_changes); drops per-event records (segment-list copies).
    Allocation decisions do not read the trace."""
    def _record(self, block, action, use_size=True):
        pass

MiB = 2 ** 20
BIG = 1 << 40


class Fake:
    def __init__(self, a):
        self.address = a


def make_sim(cap, policy):
    sim = CachingAllocator(allowed_memory_maximum=cap)
    sim._trace = LightTrace(sim._segments)
    if policy == "descending":
        state = {"lo": 1 << 46}

        def gm(size, state=state):
            state["lo"] -= size
            return Fake(state["lo"])
        sim._gpu_device.malloc = gm
        sim._gpu_device.addr_free = lambda addr: None
    return sim


def replay(stream, marks, cap, policy, inject=()):
    """stream: list of (is_alloc, size, key); marks: {name: index into stream (events before)};
    inject: [(position, size)] never-freed extra allocs placed before stream[position]."""
    sim = make_sim(cap, policy)
    dev, st, live = Device(index=0), Stream(index=0), {}
    inj = {}
    for p, s in inject:
        inj.setdefault(p, []).append(s)
    at = {}
    pos_names = {}
    for n, p in marks.items():
        pos_names.setdefault(p, []).append(n)
    peak_m0 = 0
    in_m0 = False
    try:
        for i, (a, s, k) in enumerate(stream):
            for n in pos_names.get(i, ()):
                at[n] = (sim.total_allocated_memory, sim._trace.max_usage_changes[-1])
                if n == "measure_0_begin":
                    in_m0 = True
                    peak_m0 = sim.total_allocated_memory
            for s2 in inj.get(i, ()):
                sim.malloc(dev, st, s2)
            if a:
                live[k] = sim.malloc(dev, st, s)
            else:
                b = live.pop(k, None)
                if b is not None:
                    sim.free(b)
            if in_m0 and sim.total_allocated_memory > peak_m0:
                peak_m0 = sim.total_allocated_memory
        for n in pos_names.get(len(stream), ()):
            at[n] = (sim.total_allocated_memory, sim._trace.max_usage_changes[-1])
    except MemoryError:
        return {"oom": True, "at": {n: v[0] for n, v in at.items()}}
    return {"oom": False, "m0_start_reserved": at["measure_0_begin"][0], "m0_peak_reserved": peak_m0,
            "after_load_reserved": at["model_load_end"][0],
            "max_reserved": max(sim._trace.max_segment_changes), "max_allocated": max(sim._trace.max_usage_changes)}


def gpu_stream(tdir):
    T = json.loads((tdir / "trace.json").read_text())
    ev = pickle.load(gzip.open(tdir / T["history"]["snapshot_file"], "rb"))["device_traces"][0]
    bounds = T["history"]["boundaries"]
    stream, live, na, nf, marks, bi = [], {}, 0, 0, {}, 0
    ws_pos = []

    def chk():
        nonlocal bi
        while bi < len(bounds) and na == bounds[bi]["allocation.all.allocated"] and nf == bounds[bi]["allocation.all.freed"]:
            marks.setdefault(bounds[bi]["name"], len(stream))
            bi += 1
    chk()
    nid = 0
    for e in ev:
        a = e["action"]
        if a == "alloc":
            live[e["addr"]] = nid
            stream.append((1, int(e["size"]), nid)); nid += 1; na += 1
        elif a == "free_completed":
            k = live.pop(e["addr"], None)
            stream.append((0, int(e["size"]), k)); nf += 1
        else:
            continue
        chk()
    b = {x["name"]: x for x in bounds}
    gpu = {"m0_start_reserved": b["measure_0_begin"]["reserved_bytes.all.current"],
           "m0_peak_reserved": T["measures"][0]["peak"]["reserved"],
           "m0_peak_allocated": T["measures"][0]["peak"]["allocated"],
           "after_load_reserved": b["model_load_end"]["reserved_bytes.all.current"],
           "cap": T["environment"]["cuda_total_memory"]}
    return stream, marks, gpu


def cpu_stream(cdir):
    z = np.load(cdir / "invocation_fp16_requests.npz")
    le = z["load_events"]
    stream = [(int(a), int(s), ("L", int(k))) for a, s, k in le]
    marks = {"model_load_begin": 0, "model_load_end": len(stream)}
    act, size, aid = z["action"], z["size"], z["alloc_id"]
    for r, name in enumerate(["warmup_0", "warmup_1", "warmup_2", "measure_0"]):
        marks[name + "_begin"] = len(stream)
        stream += [(int(act[j]), int(size[j]), (r, int(aid[j]))) for j in range(len(act))]
        marks[name + "_end"] = len(stream)
    res = json.loads((cdir / "result.json").read_text())
    return stream, marks, res.get("summary", {})


def window(stream, marks, name):
    return stream[marks[name + "_begin"]:marks[name + "_end"]]


def live_profile(win):
    live, cur, peak, sizes = {}, 0, 0, []
    for a, s, k in win:
        if a:
            live[k] = s; cur += s; sizes.append(s)
            peak = max(peak, cur)
        else:
            cur -= live.pop(k, 0)
    return peak, sizes


def rup(s):
    return 512 * ((s + 511) // 512) if s else 512


def diff(gw, cw):
    gp, gs = live_profile(gw)
    cp, cs = live_profile(cw)
    G, C = Counter(rup(s) for s in gs), Counter(rup(s) for s in cs)
    big = 1 * MiB
    Gb = Counter({k: v for k, v in G.items() if k >= big})
    Cb = Counter({k: v for k, v in C.items() if k >= big})
    only_g, only_c = Gb - Cb, Cb - Gb
    top = lambda c: [[k, v] for k, v in sorted(c.items(), key=lambda kv: -kv[0] * kv[1])[:8]]
    # order check on >=1MiB requests: longest common prefix of size sequence
    gseq = [rup(s) for s in gs if rup(s) >= big]
    cseq = [rup(s) for s in cs if rup(s) >= big]
    lcp = 0
    for x, y in zip(gseq, cseq):
        if x != y:
            break
        lcp += 1
    return {"gpu_n_alloc": len(gs), "cpu_n_alloc": len(cs), "gpu_bytes": sum(gs), "cpu_bytes": sum(cs),
            "gpu_peak_live": gp, "cpu_peak_live": cp,
            "big_n_gpu": len(gseq), "big_n_cpu": len(cseq), "big_seq_common_prefix": lcp,
            "big_first_mismatch": None if lcp >= min(len(gseq), len(cseq)) else {"i": lcp, "gpu": gseq[lcp], "cpu": cseq[lcp]},
            "big_only_gpu_bytes": sum(k * v for k, v in only_g.items()), "big_only_cpu_bytes": sum(k * v for k, v in only_c.items()),
            "big_only_gpu_top": top(only_g), "big_only_cpu_top": top(only_c)}


def main():
    out = Path(sys.argv[1])
    for spec in sys.argv[2:]:
        cid, gdir, cdir = spec.split("=")
        t0 = time.time()
        gs, gm, gpu = gpu_stream(Path(gdir))
        row = {"cid": cid, "gpu": gpu}
        for pol in ("stock", "descending"):
            row[f"gpu_stream_{pol}"] = replay(gs, gm, gpu["cap"], pol)
        if cdir and Path(cdir, "invocation_fp16_requests.npz").exists():
            cs, cm, csum = cpu_stream(Path(cdir))
            row["cpu_result_summary"] = {k: csum.get(k) for k in ("first_measure_peak_reserved", "first_measure_start_reserved", "after_load_reserved")}
            for pol in ("stock", "descending"):
                row[f"cpu_stream_{pol}_1TiB"] = replay(cs, cm, BIG, pol)
                row[f"cpu_stream_{pol}_cap"] = replay(cs, cm, gpu["cap"], pol)
            row["diff_load"] = diff(window(gs, gm, "model_load"), window(cs, cm, "model_load"))
            row["diff_warmup0"] = diff(window(gs, gm, "warmup_0"), window(cs, cm, "warmup_0"))
        row["wall_s"] = round(time.time() - t0, 1)
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        g = gpu["m0_start_reserved"] / MiB
        fmt = lambda r: "OOM" if r.get("oom") else f"{r['m0_start_reserved'] / MiB:.0f}"
        msg = f"{cid}: GPU {g:.0f} | gpu-stream stock {fmt(row['gpu_stream_stock'])} desc {fmt(row['gpu_stream_descending'])}"
        if "cpu_stream_stock_1TiB" in row:
            msg += (f" | cpu-stream stock {fmt(row['cpu_stream_stock_1TiB'])} desc {fmt(row['cpu_stream_descending_1TiB'])}"
                    f" desc@cap {fmt(row['cpu_stream_descending_cap'])}")
        print(msg, f"({row['wall_s']}s)", flush=True)


main()
