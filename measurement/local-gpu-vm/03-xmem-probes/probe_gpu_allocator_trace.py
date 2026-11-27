import copy
import os
from collections import Counter, defaultdict
import torch

from exp.trainer import FastRunner
from perf_estimator.config import Config

MODEL = "distilgpt2"
TRACE_FORWARDS = int(os.environ.get("TRACE_FORWARDS", "3"))
BS = 30
DEVICE = torch.device("cuda:0")

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

# C++ stack은 빼고 Python stack만 기록
torch.cuda.memory._record_memory_history(
    stacks="python",
    max_entries=200000,
)

torch.cuda.reset_peak_memory_stats()

model.to(DEVICE)
model.eval()

forward_count = 0

with torch.no_grad():
    for batch in dl:
        batch = {
            k: v.to(DEVICE)
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
        if forward_count >= TRACE_FORWARDS:
            break

torch.cuda.synchronize()

peak_alloc = torch.cuda.max_memory_allocated()
peak_res = torch.cuda.max_memory_reserved()

snapshot = torch.cuda.memory._snapshot()

torch.cuda.memory._record_memory_history(enabled=None)

traces = snapshot["device_traces"][0]

actions = Counter()
action_bytes = defaultdict(int)
streams = Counter()

for e in traces:
    action = e.get("action")
    actions[action] += 1

    size = int(e.get("size", 0) or 0)
    action_bytes[action] += size

    if "stream" in e:
        streams[e["stream"]] += 1

print("MODEL:", MODEL)
print("BATCH:", BS)
print("FORWARDS:", forward_count)
print("PEAK_ALLOCATED:", peak_alloc)
print("PEAK_RESERVED:", peak_res)

print()
print("========== ACTION COUNTS ==========")
for k, v in actions.items():
    print(k, v)

print()
print("========== ACTION BYTES ==========")
for k, v in action_bytes.items():
    print(k, v)

print()
print("========== STREAMS ==========")
for k, v in streams.items():
    print(k, v)

print()
print("========== GPU ALLOC SIZE HISTOGRAM ==========")

alloc_sizes = Counter(
    int(e.get("size", 0) or 0)
    for e in traces
    if e.get("action") == "alloc"
)

for size, count in alloc_sizes.most_common(30):
    print(size, count)

print()
print("========== GPU SEGMENT SIZE HISTOGRAM ==========")

segment_sizes = Counter(
    int(e.get("size", 0) or 0)
    for e in traces
    if e.get("action") == "segment_alloc"
)

for size, count in segment_sizes.most_common():
    print(size, count)

print()
print("========== FIRST 30 EVENTS ==========")

for i, e in enumerate(traces[:30]):
    print(
        i,
        "action=", e.get("action"),
        "addr=", e.get("addr"),
        "size=", e.get("size"),
        "stream=", e.get("stream"),
    )

print()
print("========== LAST 30 EVENTS ==========")

for i, e in enumerate(traces[-30:]):
    print(
        len(traces) - 30 + i,
        "action=", e.get("action"),
        "addr=", e.get("addr"),
        "size=", e.get("size"),
        "stream=", e.get("stream"),
    )
