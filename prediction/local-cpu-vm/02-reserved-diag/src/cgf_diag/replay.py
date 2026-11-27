"""Sequential, order-preserving replay of allocator requests into the pinned xMem CachingAllocator.

* Requests are applied strictly in `seq` order (no timestamp grouping, no malloc-before-free
  reordering). The production path (xMem AllocatorSim.simulate) groups by timestamp; that is
  reproduced separately in `time_grouped_xmem` for comparison.
* Only alloc / free requests and explicit empty_cache calls are inputs. Recorded GPU segment events
  are never injected; they are used only to compare decisions.
* GPU free_requested -> no-op; the block returns to xMem at free_completed (reuse point preserved).
* Streams are preserved as xMem Stream(index=k) (pool key includes stream index). Cross-stream event
  waits are NOT modelled (reported if present).
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

GIB = 1024 ** 3
MIB = 1024 ** 2


def verify_xmem(xmem_root, lock_path):
    lock = json.loads(Path(lock_path).read_text())
    out = {"commit": lock["commit"], "files": {}}
    ok = True
    for rel, expected in lock["sha256"].items():
        got = hashlib.sha256((Path(xmem_root) / rel).read_bytes()).hexdigest()
        out["files"][rel] = got == expected
        ok &= got == expected
    import perf_estimator.allocator.allocator as mod
    out["imported_from"] = mod.__file__
    out["imported_matches_root"] = Path(mod.__file__).resolve() == (Path(xmem_root) / "perf_estimator/allocator/allocator.py").resolve()
    out["ok"] = ok and out["imported_matches_root"]
    if not out["ok"]:
        raise RuntimeError(f"xMem source does not match lock: {out}")
    return out


def _classes():
    from perf_estimator.allocator.allocator import CachingAllocator, Trace, Device, Stream

    class RecordingTrace(Trace):
        """Keeps inherited counters; records (action, size, seg_id) instead of full segment copies."""

        def __init__(self, segments):
            super().__init__(segments)
            self.events = []

        def _record(self, block, action, use_size=True):
            self.events.append((action, block.size, block.seg_id))

    class PT26AllocSizeBoundary(CachingAllocator):
        """DIAGNOSTIC ONLY: PyTorch 2.6 get_allocation_size uses `size < kMinLargeAlloc`."""

        def get_allocation_size(self, size):
            if size <= self.kSmallSize:
                return self.kSmallBuffer
            if size < self.kMinLargeAlloc:
                return self.kLargeBuffer
            return self.kRoundLarge * ((size + self.kRoundLarge - 1) // self.kRoundLarge)

    return CachingAllocator, PT26AllocSizeBoundary, RecordingTrace, Device, Stream


POLICIES = ("xmem_pinned", "diag_pt26_alloc_size_boundary")


class Replayer:
    def __init__(self, capacity_bytes: int, policy: str = "xmem_pinned", keep_decisions: bool = True):
        if policy not in POLICIES:
            raise ValueError(policy)
        CachingAllocator, PT26, RecordingTrace, Device, Stream = _classes()
        cls = CachingAllocator if policy == "xmem_pinned" else PT26
        self.policy, self.capacity_bytes = policy, int(capacity_bytes)
        self.a = cls(allowed_memory_maximum=self.capacity_bytes)
        self.a._trace = RecordingTrace(self.a._segments)
        self.Device, self.Stream = Device, Stream
        self.device = Device(index=0)
        self.streams = {}
        self.blocks = {}
        self.pending_free = {}
        self.keep = keep_decisions
        self.decisions = []
        self.n_alloc = self.n_free = 0
        self.peak_reserved = self.peak_allocated = 0
        self.phase_stats = {}
        self.oom = None
        self.unknown_free = 0
        self.deferred_free_gap_events = 0
        self.empty_cache_calls = 0

    # --------------------------------------------------------------- state
    @property
    def reserved(self):
        return self.a.total_allocated_memory

    @property
    def allocated(self):
        return self.a._trace.max_usage_changes[-1]

    def _stream(self, s):
        key = 0 if s is None else s
        if key not in self.streams:
            self.streams[key] = self.Stream(index=len(self.streams))
        return self.streams[key]

    def _phase(self, row):
        key = (row.get("phase"), row.get("invocation"))
        st = self.phase_stats.get(key)
        if st is None:
            st = self.phase_stats[key] = {"phase": key[0], "invocation": key[1], "reserved_at_start": self.reserved,
                                          "allocated_at_start": self.allocated, "peak_reserved": self.reserved,
                                          "peak_allocated": self.allocated, "requests": 0, "new_segments": 0,
                                          "new_segment_bytes": 0}
        return st

    def _update(self, st):
        r, a = self.reserved, self.allocated
        self.peak_reserved = max(self.peak_reserved, r)
        self.peak_allocated = max(self.peak_allocated, a)
        st["peak_reserved"] = max(st["peak_reserved"], r)
        st["peak_allocated"] = max(st["peak_allocated"], a)
        st["reserved_at_end"], st["allocated_at_end"] = r, a

    # --------------------------------------------------------------- apply
    def apply(self, row):
        if self.oom is not None:
            return False
        act = row["action"]
        st = self._phase(row)
        if act == "alloc":
            ev0 = len(self.a._trace.events)
            try:
                block = self.a.malloc(self.device, self._stream(row.get("stream")), int(row["size"]))
            except MemoryError as exc:
                self.oom = {"seq": row["seq"], "size": row["size"], "error": str(exc), "reserved": self.reserved,
                            "allocated": self.allocated, "phase": row.get("phase"), "invocation": row.get("invocation")}
                return False
            new = self.a._trace.events[ev0:]
            seg_alloc = [s for (k, s, _) in new if k == "segment_alloc"]
            seg_free = [s for (k, s, _) in new if k == "segment_free"]
            self.blocks[row["alloc_id"]] = block
            self.n_alloc += 1
            st["requests"] += 1
            st["new_segments"] += len(seg_alloc)
            st["new_segment_bytes"] += sum(seg_alloc)
            if self.keep:
                self.decisions.append((row["seq"], row["alloc_id"], int(row["size"]), block.size, sum(seg_alloc),
                                       sum(seg_free), self.reserved, self.allocated, block.seg_id))
            if len(self.a._trace.events) > 100000:
                self.a._trace.events.clear()
        elif act in ("free", "free_completed"):
            block = self.blocks.pop(row["alloc_id"], None) if row.get("alloc_id") is not None else None
            if block is None:
                self.unknown_free += 1
                return True
            if act == "free_completed" and row["alloc_id"] in self.pending_free:
                gap = row["seq"] - self.pending_free.pop(row["alloc_id"])
                if gap > 1:
                    self.deferred_free_gap_events += 1
            self.a.free(block)
            self.n_free += 1
        elif act == "free_requested":
            if row.get("alloc_id") is not None:
                self.pending_free[row["alloc_id"]] = row["seq"]
        elif act == "empty_cache_call":
            self.a.release_cached_blocks()
            self.empty_cache_calls += 1
        self._update(st)
        return True

    def place_resident(self, sizes, phase="resident_synthetic"):
        """Baseline-equivalent synthetic residency: each storage malloc'd in order, never freed."""
        for i, size in enumerate(sizes):
            if size:
                self.apply({"seq": -len(sizes) + i, "action": "alloc", "size": size, "stream": 0,
                            "alloc_id": ("resident", i), "phase": phase, "invocation": None})

    def run(self, rows):
        for row in rows:
            if row["action"] in ("alloc", "free", "free_completed", "free_requested", "empty_cache_call"):
                if not self.apply(row):
                    break
        return self.summary()

    def summary(self):
        return {"policy": self.policy, "capacity_bytes": self.capacity_bytes,
                "peak_reserved_bytes": self.peak_reserved, "peak_allocated_bytes": self.peak_allocated,
                "final_reserved_bytes": self.reserved, "final_allocated_bytes": self.allocated,
                "requests": self.n_alloc, "frees": self.n_free, "live_blocks_at_end": len(self.blocks),
                "unknown_frees": self.unknown_free, "deferred_free_gaps": self.deferred_free_gap_events,
                "empty_cache_calls": self.empty_cache_calls, "segments_at_end": len(self.a._segments),
                "oom": self.oom,
                "phases": list(self.phase_stats.values())}


DECISION_FIELDS = ("seq", "alloc_id", "request_bytes", "block_bytes", "new_segment_bytes", "released_segment_bytes",
                   "reserved_after", "allocated_after", "segment_id")


# ----------------------------------------------------------------- production-equivalent path
def time_grouped_xmem(rows, resident_sizes, capacity_gb, *, first_region_ts):
    """Rebuild MemoryBlocks exactly as cpu_tensor_trace.trace_and_estimate does and run the pinned
    AllocatorSim (timestamp grouping, malloc-before-free inside a group) via the predictor's
    inference_trace_runtime.simulate. Returns (allocated_peak, reserved_peak)."""
    import uuid
    from perf_estimator.profiler import MemoryBlock, CpuInstantNode
    from perf_estimator.config import Config
    from inference_trace_runtime import simulate

    def node(ts, size, addr):
        return CpuInstantNode({"ph": "i", "cat": "cpu_instant_event", "s": "t", "name": "[memory]", "pid": 0, "tid": 0,
                               "ts": ts, "args": {"Bytes": size, "Addr": addr, "Device Id": -1, "Device Type": 0, "Ev Idx": 0}})

    allocs = {}
    blocks = []
    for r in rows:
        if r["action"] == "alloc":
            b = MemoryBlock(node(r["t"], r["size"], r["addr"]))
            allocs[r["alloc_id"]] = b
            blocks.append(b)
        elif r["action"] in ("free", "free_completed") and r["alloc_id"] in allocs:
            allocs.pop(r["alloc_id"]).set_free_node(node(r["t"], -r["size"], r["addr"]))
    for index, size in enumerate(resident_sizes):
        if size:
            blocks.append(MemoryBlock(node(first_region_ts - 1, size, -index - 1)))
    blocks.sort(key=lambda b: b.alloc_time)
    result = simulate(blocks, capacity_gb, Config(save2tmp=True, run_id=uuid.uuid4().hex), record_history=False)
    return max(result._trace.max_usage_changes), max(result._trace.max_segment_changes)


def timestamp_collision_report(rows):
    """How many request events share a timestamp with another event (what AllocatorSim regroups)."""
    by_t = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["action"] == "alloc":
            by_t[r["t"]][0] += 1
        elif r["action"] in ("free", "free_completed"):
            by_t[r["t"]][1] += 1
    groups = [v for v in by_t.values() if sum(v) > 1]
    mixed = [v for v in groups if v[0] and v[1]]
    return {"distinct_timestamps": len(by_t), "multi_event_groups": len(groups), "groups_with_alloc_and_free": len(mixed),
            "events_in_mixed_groups": sum(sum(v) for v in mixed)}
