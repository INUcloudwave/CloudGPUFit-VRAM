import copy
import torch

from exp.trainer import FastRunner
from perf_estimator.config import Config
from perf_estimator.allocator.allocator import (
    CachingAllocator,
    Device,
    Stream,
)

MODEL = "distilgpt2"
BS = 30
GPU = torch.device("cuda:0")

conf = Config(save2tmp=False)

runner = FastRunner(
    model_name=MODEL,
    batch_size=BS,
    optimiser="SGD",
    gpu_id=0,
    config=conf,
)

model = copy.deepcopy(runner.model_preparer.model)
dl = copy.deepcopy(runner.model_preparer.dl)

torch.cuda.empty_cache()
torch.cuda.synchronize()

torch.cuda.memory._record_memory_history(
    stacks="python",
    max_entries=200000,
)

torch.cuda.reset_peak_memory_stats()

model.to(GPU)
model.eval()

forward_count = 0

with torch.no_grad():
    for batch in dl:
        batch = {
            k: v.to(GPU)
            for k, v in batch.items()
            if k != "labels"
        }

        if (
            batch["input_ids"].numel() == 0
            or batch["input_ids"].shape[-1] == 0
        ):
            continue

        model(**batch, use_cache=False)

        forward_count += 1
        if forward_count >= 3:
            break

torch.cuda.synchronize()

real_peak_alloc = torch.cuda.max_memory_allocated()
real_peak_reserved = torch.cuda.max_memory_reserved()

snapshot = torch.cuda.memory._snapshot()
torch.cuda.memory._record_memory_history(enabled=None)

events = snapshot["device_traces"][0]

# ------------------------------------------------------------
# 1. free_requested -> free_completed 지연 정도 분석
# ------------------------------------------------------------
pending = {}
alloc_counter = 0

pair_count = 0
immediate_pairs = 0
pairs_with_alloc_between = 0
max_event_gap = 0
max_alloc_between = 0

for i, e in enumerate(events):
    action = e.get("action")
    addr = e.get("addr")

    if action == "alloc":
        alloc_counter += 1

    elif action == "free_requested":
        pending[addr] = (i, alloc_counter)

    elif action == "free_completed":
        if addr in pending:
            request_i, request_alloc_count = pending.pop(addr)

            event_gap = i - request_i
            alloc_between = alloc_counter - request_alloc_count

            pair_count += 1

            if event_gap == 1:
                immediate_pairs += 1

            if alloc_between > 0:
                pairs_with_alloc_between += 1

            max_event_gap = max(max_event_gap, event_gap)
            max_alloc_between = max(max_alloc_between, alloc_between)


# ------------------------------------------------------------
# 2. 실제 GPU alloc/free event를 xMem allocator에 그대로 replay
# ------------------------------------------------------------
sim = CachingAllocator(
    allowed_memory_maximum=24 * 1024**3
)

device = Device(index=0)
streams = {}
active = {}

alloc_events = 0
free_events = 0
missing_frees = 0

for e in events:
    action = e.get("action")

    if action == "alloc":
        addr = e["addr"]
        size = int(e["size"])
        stream_id = int(e.get("stream", 0))

        if stream_id not in streams:
            streams[stream_id] = Stream(index=stream_id)

        active[addr] = sim.malloc(
            device,
            streams[stream_id],
            size,
        )

        alloc_events += 1

    elif action == "free_completed":
        addr = e["addr"]

        if addr not in active:
            missing_frees += 1
            continue

        sim.free(active.pop(addr))
        free_events += 1


sim_peak_alloc = max(sim._trace.max_usage_changes)
sim_peak_reserved = max(sim._trace.max_segment_changes)

print()
print("========== REAL GPU ==========")
print("forwards:", forward_count)
print("peak allocated:", real_peak_alloc)
print("peak reserved :", real_peak_reserved)

print()
print("========== FREE DELAY ==========")
print("free pairs:", pair_count)
print("immediate pairs:", immediate_pairs)
print("pairs with alloc between:", pairs_with_alloc_between)
print("max event gap:", max_event_gap)
print("max allocs between:", max_alloc_between)

print()
print("========== XMEM REPLAY ==========")
print("alloc events:", alloc_events)
print("free events:", free_events)
print("missing frees:", missing_frees)
print("active blocks left:", len(active))

print("sim peak allocated:", sim_peak_alloc)
print("sim peak reserved :", sim_peak_reserved)

print()
print("========== DIFFERENCE ==========")
print(
    "allocated diff:",
    sim_peak_alloc - real_peak_alloc
)
print(
    "reserved diff:",
    sim_peak_reserved - real_peak_reserved
)
print(
    "reserved error %:",
    100.0 * (
        sim_peak_reserved - real_peak_reserved
    ) / real_peak_reserved
)
