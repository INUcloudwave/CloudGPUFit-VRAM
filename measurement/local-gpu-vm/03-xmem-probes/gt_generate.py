import json, sys, copy, torch
sys.path.insert(0, '/work/xMem')
from predict_vram import load_model, build_loader

model_name, batch, dtype, seq_len, max_new = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])

model, kind, cfg = load_model(model_name)
assert kind == 'llm', 'generate mode는 LLM 전용'
loader = build_loader('llm', batch, seq_len=seq_len, vocab_size=getattr(cfg, 'vocab_size', 30000))

dt = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[dtype]
m = copy.deepcopy(model).to(dt).cuda().eval()

torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    batch_data = next(iter(loader))
    b = {k: v.cuda() for k, v in batch_data.items() if k != 'labels'}
    m.generate(**b, max_new_tokens=max_new, do_sample=False, use_cache=True)
    torch.cuda.synchronize()

print(json.dumps({
    'model': model_name, 'batch_size': batch, 'seq_len': seq_len,
    'max_new_tokens': max_new, 'dtype': dtype,
    'allocated_peak': torch.cuda.max_memory_allocated(),
    'reserved_peak': torch.cuda.max_memory_reserved(),
}, indent=2))
