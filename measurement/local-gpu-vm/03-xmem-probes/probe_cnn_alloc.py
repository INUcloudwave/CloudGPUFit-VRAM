import copy
import json
import sys
import torch

from exp.trainer import FastRunner
from perf_estimator.config import Config

model_name = sys.argv[1]
batch_size = int(sys.argv[2])
gpu_id = 0
device = torch.device(f"cuda:{gpu_id}")

conf = Config(save2tmp=False)

runner = FastRunner(
    model_name=model_name,
    batch_size=batch_size,
    optimiser="SGD",
    gpu_id=gpu_id,
    config=conf,
)

model = copy.deepcopy(runner.model_preparer.model)
dl = copy.deepcopy(runner.model_preparer.dl)

param_bytes = sum(
    x.numel() * x.element_size()
    for x in model.parameters()
)
buffer_bytes = sum(
    x.numel() * x.element_size()
    for x in model.buffers()
)
raw_model_bytes = param_bytes + buffer_bytes


def mem():
    torch.cuda.synchronize()
    s = torch.cuda.memory_stats()
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "requested_current": s.get("requested_bytes.all.current"),
        "allocated_current_stat": s.get("allocated_bytes.all.current"),
        "requested_peak": s.get("requested_bytes.all.peak"),
        "allocated_peak_stat": s.get("allocated_bytes.all.peak"),
    }


torch.cuda.empty_cache()
torch.cuda.synchronize()

baseline = mem()

# 1. 모델만 GPU로 이동
model = model.to(device)
model.eval()
after_model = mem()

# 2. 첫 입력 batch를 GPU로 이동
inputs, labels = next(iter(dl))

input_raw_bytes = inputs.numel() * inputs.element_size()

inputs_gpu = inputs.to(device)
after_input = mem()

# 3. forward peak 측정
torch.cuda.reset_peak_memory_stats()

with torch.no_grad():
    output = model(inputs_gpu)

torch.cuda.synchronize()

after_forward = mem()
peak_allocated = torch.cuda.max_memory_allocated()
peak_reserved = torch.cuda.max_memory_reserved()

result = {
    "model": model_name,
    "batch_size": batch_size,

    "raw": {
        "param_bytes": param_bytes,
        "buffer_bytes": buffer_bytes,
        "raw_model_bytes": raw_model_bytes,
        "input_raw_bytes": input_raw_bytes,
    },

    "baseline": baseline,
    "after_model": after_model,
    "after_input": after_input,
    "after_forward": after_forward,

    "derived": {
        "actual_model_alloc":
            after_model["allocated"] - baseline["allocated"],

        "model_over_raw":
            (after_model["allocated"] - baseline["allocated"])
            - raw_model_bytes,

        "actual_input_alloc":
            after_input["allocated"] - after_model["allocated"],

        "input_over_raw":
            (after_input["allocated"] - after_model["allocated"])
            - input_raw_bytes,

        "forward_extra_peak":
            peak_allocated - after_input["allocated"],

        "peak_allocated": peak_allocated,
        "peak_reserved": peak_reserved,
    },
}

print(json.dumps(result, indent=2))
