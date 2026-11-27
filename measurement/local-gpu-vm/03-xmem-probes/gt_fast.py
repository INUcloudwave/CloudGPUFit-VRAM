import json, time, sys
import torch, GPUtil, uuid, copy
from exp.trainer import FastRunner, HostMonitorPlugin
from exp.trainer.trainer import ModelTrainer
from perf_estimator.config import Config
from utils import search_nvml_file

import os

cudnn_mode = os.environ.get("CUDNN_MODE", "default")

if cudnn_mode == "default":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

elif cudnn_mode == "deterministic":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

elif cudnn_mode == "off":
    torch.backends.cudnn.enabled = False

else:
    raise ValueError(f"Unknown CUDNN_MODE: {cudnn_mode}")

print(
    f"CUDNN_MODE={cudnn_mode}",
    f"enabled={torch.backends.cudnn.enabled}",
    f"benchmark={torch.backends.cudnn.benchmark}",
    f"deterministic={torch.backends.cudnn.deterministic}",
)

model, opt, bs = sys.argv[1], sys.argv[2], int(sys.argv[3])
gpu_id = 0
conf = Config(save2tmp=False)
runner = FastRunner(model_name=model, batch_size=bs, optimiser=opt,
                    gpu_id=gpu_id, config=conf)

g_config = conf.model_copy(deep=True)
g_config.task_id = f"GPU_{uuid.uuid4().hex[:3]}"
runner.set_fraction_gpu_memory(None)
trainer = ModelTrainer(
    model=copy.deepcopy(runner.model_preparer.model),
    data_loader=copy.deepcopy(runner.model_preparer.dl),
    on_cpu=False, gpu_id=gpu_id, iterations=0,
    optimiser=runner.model_preparer.optimiser, config=g_config,
    plugins=[HostMonitorPlugin(interval_ms=10, config=g_config,
                               gpu_enable=True, cpu_enable=False,
                               network_enable=False)],
    zero_grad_mode=conf.trainer.zero_out,
)
torch.cuda.reset_peak_memory_stats()
_, oom = trainer.train(is_transformer=runner.model_preparer.is_transformer)
reserved_peak = torch.cuda.max_memory_reserved()
allocated_peak = torch.cuda.max_memory_allocated()

torch.cuda.empty_cache(); time.sleep(2)
framework_mem = int(GPUtil.getGPUs()[gpu_id].memoryUsed * 1024**2)
with open(search_nvml_file(g_config.result_dir)[-1]) as f:
    data = json.load(f)
usage, start = [], None
for i, rec in enumerate(data["records"]):
    used = rec["HostGPUs"][str(gpu_id)]["memory"]["used"]
    if i == 0: start = used
    usage.append(used - start)

out = {"model": model, "optimiser": opt, "batch_size": bs, "oom": oom,
       "nvml_peak": max(usage), "reserved_peak": reserved_peak,
       "allocated_peak": allocated_peak, "framework_mem": framework_mem,
       "samples": len(usage), "result_dir": str(g_config.result_dir)}
print(json.dumps(out, indent=2))
with open("/work/gt_result.json", "w") as f: json.dump(out, f, indent=2)
