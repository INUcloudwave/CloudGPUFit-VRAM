"""Normalized allocator-request events and importers (CPU chrome trace, GPU memory snapshot).

Schema (one JSON object per line, ordered by `seq`):
  seq        int   original order after import (never re-sorted by timestamp)
  src        str   "cpu_profiler" | "gpu_snapshot"
  action     str   alloc | free_requested | free_completed | free (CPU: request+completion are one event)
                   segment_alloc | segment_free | segment_map | segment_unmap | oom | snapshot
                   empty_cache_call (explicit API call recorded by the collector, not inferred)
  addr       int   device/host address as recorded (identity only; NOT used for xMem placement)
  size       int   bytes. alloc/free*: original request size (PyTorch 2.6 records orig_size /
                   requested_size; CPU profiler records requested nbytes). segment_*: segment size.
  stream     int|None   CUDA stream handle as recorded (CPU: 0)
  t          float|None CPU: chrome ts (us, ns fraction). GPU: time_us.
  order_key  int|None   CPU: kineto 'Ev Idx'. GPU: position in device_traces.
  alloc_id   int|None   generation id (address reuse safe) linking alloc -> free_* events
  phase      str|None   phase label from boundary records
  invocation int|None   invocation index inside the phase
Fields that the source does not provide are None ("unknown"), never guessed.
"""
from __future__ import annotations

import gzip
import io
import json
import pickle
from pathlib import Path
from typing import Iterable, Iterator

REQUEST_ACTIONS = {"alloc", "free_requested", "free_completed", "free"}
SEGMENT_ACTIONS = {"segment_alloc", "segment_free", "segment_map", "segment_unmap"}


def open_text(path, mode="rt"):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8", compresslevel=1) if "w" in mode else gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def write_jsonl(path, rows: Iterable[dict]):
    n = 0
    with open_text(path, "wt") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
            n += 1
    return n


def read_jsonl(path) -> Iterator[dict]:
    with open_text(path, "rt") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# ----------------------------------------------------------------------------- pairing
class Pairer:
    """Assigns alloc_id generations; detects address reuse while live and unmatched frees."""

    def __init__(self):
        self.live = {}          # (stream_or_None, addr) -> (alloc_id, size)
        self.next_id = 0
        self.unmatched_frees = 0
        self.size_mismatch = 0
        self.reuse_while_live = 0

    def alloc(self, addr, size, stream=None):
        key = addr
        if key in self.live:
            self.reuse_while_live += 1
        aid = self.next_id
        self.next_id += 1
        self.live[key] = (aid, size)
        return aid

    def lookup(self, addr, size):
        item = self.live.get(addr)
        if item is None:
            self.unmatched_frees += 1
            return None
        if size is not None and item[1] != size:
            self.size_mismatch += 1
        return item[0]

    def release(self, addr, size):
        aid = self.lookup(addr, size)
        if aid is not None:
            del self.live[addr]
        return aid


# ----------------------------------------------------------------------------- CPU importer
def cpu_memory_events_from_chrome(trace_path, region_prefix: str | None, *, iter_trace_events):
    """Return (regions, memory_events) from a torch.profiler chrome trace (CPU device only).

    `iter_trace_events` is the predictor's streaming parser (inference_trace_runtime.iter_trace_events)
    so very large traces are not loaded with json.load.
    regions: list of (name, ts_begin, ts_end) for user_annotation events whose name starts with
    region_prefix (None -> all user annotations).
    """
    regions, mem = [], []
    for e in iter_trace_events(trace_path):
        cat, name = e.get("cat"), e.get("name")
        if cat == "user_annotation" and (region_prefix is None or str(name).startswith(region_prefix)):
            regions.append((name, e["ts"], e["ts"] + e.get("dur", 0)))
        elif name == "[memory]" and cat == "cpu_instant_event":
            a = e.get("args", {})
            if a.get("Device Type") == 0 and a.get("Bytes", 0) != 0:
                mem.append({"ts": e["ts"], "ev_idx": a.get("Ev Idx"), "bytes": a["Bytes"], "addr": a["Addr"],
                            "tid": e.get("tid"), "total_allocated": a.get("Total Allocated")})
    regions.sort(key=lambda r: r[1])
    return regions, mem


def order_report(mem):
    """Compare Ev Idx order with timestamp order (both are reported; Ev Idx is used)."""
    by_idx = sorted(mem, key=lambda m: (m["ev_idx"], m["ts"]))
    inversions = sum(1 for a, b in zip(by_idx, by_idx[1:]) if b["ts"] < a["ts"])
    ts_values = [m["ts"] for m in mem]
    dup_ts = len(ts_values) - len(set(ts_values))
    idx_missing = sum(1 for m in mem if m["ev_idx"] is None)
    tids = sorted({m["tid"] for m in mem if m["tid"] is not None})
    return {"memory_events": len(mem), "ev_idx_missing": idx_missing,
            "ts_inversions_in_ev_idx_order": inversions, "duplicate_ts_events": dup_ts, "tids": tids}


def normalize_cpu(mem, regions, *, region_filter=True, phase_of=None, start_seq=0, pairer=None):
    """CPU memory events -> normalized rows in Ev Idx order.

    Selection rule identical to the production predictor: an allocation is a request only if it
    occurs inside a selected region; its free is matched wherever it occurs. Frees of allocations
    made outside (before) the regions are reported as unmatched and not replayed.
    """
    pairer = pairer or Pairer()
    rows, seq = [], start_seq
    spans = [(b, e, name) for name, b, e in regions]

    def region_name(ts):
        for b, e, name in spans:
            if b <= ts <= e:
                return name
        return None

    skipped_outside = 0
    for m in sorted(mem, key=lambda m: (m["ev_idx"] if m["ev_idx"] is not None else -1, m["ts"])):
        size = m["bytes"]
        if size > 0:
            name = region_name(m["ts"])
            if region_filter and name is None:
                skipped_outside += 1
                continue
            aid = pairer.alloc(m["addr"], size)
            ph, inv = phase_of(name) if phase_of else (name, None)
            rows.append(dict(seq=seq, src="cpu_profiler", action="alloc", addr=m["addr"], size=size, stream=0,
                             t=m["ts"], order_key=m["ev_idx"], alloc_id=aid, phase=ph, invocation=inv))
        else:
            aid = pairer.release(m["addr"], -size)
            if aid is None:
                continue
            name = region_name(m["ts"])
            ph, inv = phase_of(name) if phase_of else (name, None)
            rows.append(dict(seq=seq, src="cpu_profiler", action="free", addr=m["addr"], size=-size, stream=0,
                             t=m["ts"], order_key=m["ev_idx"], alloc_id=aid, phase=ph, invocation=inv))
        seq += 1
    stats = {"skipped_alloc_outside_regions": skipped_outside, "unmatched_frees": pairer.unmatched_frees,
             "free_size_mismatch": pairer.size_mismatch, "addr_reuse_while_live": pairer.reuse_while_live,
             "live_at_end": len(pairer.live)}
    return rows, stats, pairer


# ----------------------------------------------------------------------------- GPU importer
class _SafeUnpickler(pickle.Unpickler):
    """Snapshots are plain dict/list/str/int/float; refuse any global (no code execution)."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing global {module}.{name} in memory snapshot")


def load_snapshot(path):
    path = Path(path)
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    return _SafeUnpickler(io.BytesIO(data)).load()


def normalize_gpu_snapshot(snapshot, *, device=0, boundaries=None):
    """GPU snapshot device_traces -> normalized rows in recorded order.

    boundaries: optional list of dicts written by the collector at each phase/invocation start, each
      {"phase", "invocation", "counters": {"alloc": memory_stats["allocation.all.allocated"],
        "free": memory_stats["allocation.all.freed"], "segment_alloc": memory_stats["segment.all.allocated"],
        "segment_free": memory_stats["segment.all.freed"]}, "time_us": ...}
      taken after torch.cuda.synchronize(). A boundary is placed at the first trace position where the
      running action counts equal its counters (exact when recording started before the first CUDA
      allocation). time_us is only a cross-check. Unplaceable boundaries are reported.
    """
    traces = snapshot.get("device_traces", [])
    if device >= len(traces):
        raise ValueError("snapshot has no trace for device %d" % device)
    pairer = Pairer()
    rows = []
    pending = list(boundaries or [])
    counts = {"alloc": 0, "free": 0, "segment_alloc": 0, "segment_free": 0}
    key_of = {"alloc": "alloc", "free_completed": "free", "segment_alloc": "segment_alloc", "segment_free": "segment_free"}
    placed, ph, inv = [], None, None

    def advance(position):
        nonlocal ph, inv
        while pending and all(counts[k] >= int(pending[0]["counters"][k]) for k in counts):
            b = pending.pop(0)
            exact = all(counts[k] == int(b["counters"][k]) for k in counts)
            placed.append({"phase": b.get("phase"), "invocation": b.get("invocation"), "position": position, "exact": exact})
            ph, inv = b.get("phase"), b.get("invocation")

    frames = []
    advance(0)
    for i, e in enumerate(traces[device]):
        action = e.get("action")
        addr, size, stream = e.get("addr"), e.get("size"), e.get("stream")
        aid = None
        if action == "alloc":
            aid = pairer.alloc(addr, size)
        elif action == "free_requested":
            aid = pairer.lookup(addr, size)
        elif action == "free_completed":
            aid = pairer.release(addr, size)
        rows.append(dict(seq=i, src="gpu_snapshot", action=action, addr=addr, size=size, stream=stream,
                         t=e.get("time_us"), order_key=i, alloc_id=aid, phase=ph, invocation=inv))
        if action in key_of:
            counts[key_of[action]] += 1
        advance(i + 1)
        if e.get("frames"):
            frames.append({"seq": i, "frames": e["frames"][:12]})
    stats = {"entries": len(rows), "unmatched_frees": pairer.unmatched_frees,
             "free_size_mismatch": pairer.size_mismatch, "addr_reuse_while_live": pairer.reuse_while_live,
             "live_at_end": len(pairer.live),
             "actions": {a: sum(1 for r in rows if r["action"] == a) for a in sorted({str(r["action"]) for r in rows})},
             "boundaries_placed": placed, "boundaries_unplaced": pending}
    return rows, stats, frames


def gpu_reserved_timeline(rows, initial_reserved=0):
    """Observed reserved bytes after each GPU event = initial + segment_alloc - segment_free (+map/unmap)."""
    cur = initial_reserved
    out = []
    for r in rows:
        if r["action"] in ("segment_alloc", "segment_map"):
            cur += r["size"]
        elif r["action"] in ("segment_free", "segment_unmap"):
            cur -= r["size"]
        out.append(cur)
    return out
