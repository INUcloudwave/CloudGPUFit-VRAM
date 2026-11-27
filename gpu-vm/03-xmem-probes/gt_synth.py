import json, sys, copy, torch
sys.path.insert(0, "/work/xMem")
from predict_vram import load_model, build_loader

model_name, batch, dtype = sys.argv[1], int(sys.argv[2]), sys.argv[3]
size = int(sys.argv[4])          # llm이면 seq_len, vision이면 resolution

model, kind, cfg = load_model(model_name)
loader = (build_loader("llm", batch, seq_len=size,
                       vocab_size=getattr(cfg, "vocab_size", 30000))
          if kind == "llm" else
          build_loader("vision", batch, resolution=size))

dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
m = copy.deepcopy(model).to(dt).cuda().eval()

torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    for i, batch_data in enumerate(loader):
        if kind == "llm":
            b = {k: v.cuda() for k, v in batch_data.items()}
            m(**b, use_cache=False)
        else:
            m(batch_data[0].cuda().to(dt))
        torch.cuda.synchronize()
        if i == 2:
            break

print(json.dumps({
    "model": model_name, "type": kind, "batch_size": batch,
    "size": size, "dtype": dtype,
    "allocated_peak": torch.cuda.max_memory_allocated(),
    "reserved_peak": torch.cuda.max_memory_reserved(),
}, indent=2))
