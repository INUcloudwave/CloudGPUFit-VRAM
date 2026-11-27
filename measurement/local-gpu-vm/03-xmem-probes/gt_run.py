import json, time, sys
import torch, GPUtil
from exp.trainer import FastRunner
from perf_estimator.config import Config
from utils import search_nvml_file, search_snapshot_file

model, opt, bs = sys.argv[1], sys.argv[2], int(sys.argv[3])
gpu_id = 0

conf = Config(save2tmp=False)
runner = FastRunner(model_name=model, batch_size=bs, optimiser=opt,
                    gpu_id=gpu_id, config=conf)
g_config, oom = runner.train_on_gpu()
torch.cuda.empty_cache(); time.sleep(2)
framework_mem = int(GPUtil.getGPUs()[gpu_id].memoryUsed * 1024**2)

# NVML delta peak (논문식)
with open(search_nvml_file(g_config.result_dir)[-1]) as f:
    data = json.load(f)
usage, start = [], None
for i, rec in enumerate(data["records"]):
    used = rec["HostGPUs"][str(gpu_id)]["memory"]["used"]
    if i == 0: start = used
    usage.append(used - start)
nvml_peak = max(usage)

# snapshot 세그먼트 피크 (정밀)
from exp.snapshot import SnapshotAnalyser
seg, _ = SnapshotAnalyser(str(search_snapshot_file(g_config.result_dir)[-1])
                          ).gpu_and_segment_max_memory_changes_data()
snap_peak = max(seg)

out = {"model": model, "optimiser": opt, "batch_size": bs, "oom": oom,
       "nvml_peak": nvml_peak, "snap_peak": snap_peak,
       "framework_mem": framework_mem,
       "result_dir": str(g_config.result_dir)}
print(json.dumps(out, indent=2))
with open("/work/gt_result.json", "w") as f: json.dump(out, f, indent=2)
