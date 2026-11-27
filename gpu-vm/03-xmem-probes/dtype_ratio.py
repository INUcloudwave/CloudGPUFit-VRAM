import json, sys, copy, torch
from exp.trainer import FastRunner
from perf_estimator.config import Config

model, bs = sys.argv[1], int(sys.argv[2])
conf = Config(save2tmp=False)
runner = FastRunner(model_name=model, batch_size=bs, optimiser='SGD', config=conf)
is_tf = runner.model_preparer.is_transformer
base = runner.model_preparer.model
dl = runner.model_preparer.dl
batch = next(iter(dl))

res = {}
for name, dt in [("fp32", torch.float32), ("fp16", torch.float16),
                 ("bf16", torch.bfloat16)]:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m = copy.deepcopy(base).to(dt).cuda().eval()
    try:
        with torch.no_grad():
            for _ in range(2):
                if is_tf:
                    b = {k: v.cuda() for k, v in batch.items() if k != "labels"}
                    if b["input_ids"].shape[-1] == 0: continue
                    m(**b, use_cache=False)
                else:
                    x = batch[0].cuda().to(dt)
                    m(x)
                torch.cuda.synchronize()
        res[name] = {"allocated": torch.cuda.max_memory_allocated(),
                     "reserved": torch.cuda.max_memory_reserved()}
    except Exception as e:
        res[name] = {"error": str(e)[:120]}
    del m; torch.cuda.empty_cache()

if "allocated" in res.get("fp32", {}):
    b32 = res["fp32"]["allocated"]
    for k in ("fp16", "bf16"):
        if "allocated" in res.get(k, {}):
            res[k]["ratio_vs_fp32"] = round(res[k]["allocated"] / b32, 4)
out = {"model": model, "batch_size": bs, "results": res}
print(json.dumps(out, indent=2))
