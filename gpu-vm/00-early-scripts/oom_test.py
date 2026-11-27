import argparse, json, os, sys, time, torch
GIB = 1024**3

def is_oom(e):
    for c in (getattr(torch,"OutOfMemoryError",None), getattr(torch.cuda,"OutOfMemoryError",None)):
        if c and isinstance(e,c): return True
    return isinstance(e,RuntimeError) and "out of memory" in str(e).lower()

def load(mdl, **kw):
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(mdl, dtype=torch.float16, **kw)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(mdl, torch_dtype=torch.float16, **kw)

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--mode", choices=["strict","offload"], required=True)
p.add_argument("--new-tokens", type=int, default=64)
p.add_argument("--json", default=None)
a = p.parse_args()

from transformers import AutoTokenizer
d = torch.cuda.get_device_properties(0)
print(f"[env] {d.name} VRAM={d.total_memory/GIB:.2f} GiB torch={torch.__version__}")
print(f"[cfg] {a.model} mode={a.mode}\n" + "-"*60)

kw = dict(low_cpu_mem_usage=True)
if a.mode == "offload":
    os.makedirs("./_offload", exist_ok=True)
    kw.update(device_map="auto", max_memory={0:"22GiB","cpu":"38GiB"}, offload_folder="./_offload")
else:
    kw.update(device_map={"":0})

R = {"mode":a.mode, "model":a.model, "gpu":d.name, "vram_gib":round(d.total_memory/GIB,2)}
def dump(v, **x):
    R.update(verdict=v, **x)
    if a.json: json.dump(R, open(a.json,"w"), indent=2, ensure_ascii=False)

tok = AutoTokenizer.from_pretrained(a.model)
t0 = time.perf_counter()
try:
    m = load(a.model, **kw)
except Exception as e:
    if is_oom(e):
        print(f"\n>>> LOAD_OOM ({time.perf_counter()-t0:.1f}s)\n    {str(e).splitlines()[0]}")
        print("    가중치가 안 들어감. 프로세스 종료. GPU/드라이버는 정상.")
        dump("LOAD_OOM", error=str(e).splitlines()[0]); sys.exit(1)
    raise
load_s = time.perf_counter()-t0
wb = sum(x.numel()*x.element_size() for x in m.parameters())
place = {}
for _,v in getattr(m,"hf_device_map",{"":"cuda:0"}).items():
    place[str(v)] = place.get(str(v),0)+1
off = any(k not in ("0","cuda:0","cuda") for k in place)
print(f"[load] {load_s:.1f}s weights={wb/GIB:.2f} GiB reserved={torch.cuda.memory_reserved()/GIB:.2f} GiB")
print(f"[load] placement={place}")

ids = tok("Explain why a GPU runs out of memory, in one paragraph.", return_tensors="pt").to("cuda")
try:
    with torch.inference_mode(): m.generate(**ids, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    with torch.inference_mode(): out = m.generate(**ids, max_new_tokens=a.new_tokens, do_sample=False)
    torch.cuda.synchronize(); gen_s = time.perf_counter()-t1
except Exception as e:
    if is_oom(e):
        print(f"\n>>> RUNTIME_OOM\n    {str(e).splitlines()[0]}")
        print("    로딩은 됐으나 KV cache/activation 증가로 추론 중 사망.")
        dump("RUNTIME_OOM", weights_gib=round(wb/GIB,2), error=str(e).splitlines()[0]); sys.exit(1)
    raise

n = out.shape[1]-ids["input_ids"].shape[1]
v = "OFFLOAD_OK" if off else "FIT_OK"
print(f"\n>>> {v}\n    {n} tok / {gen_s:.2f}s = {n/gen_s:.2f} tok/s")
print(f"    peak_gpu_alloc={torch.cuda.max_memory_allocated()/GIB:.2f} GiB")
if off: print("    안 죽음. 일부 레이어가 CPU에 있어 매 토큰 PCIe 전송 발생.")
dump(v, weights_gib=round(wb/GIB,2), load_s=round(load_s,2),
     tok_per_s=round(n/gen_s,2), peak_gib=round(torch.cuda.max_memory_allocated()/GIB,2), placement=place)
print(f"\n저장: {a.json}" if a.json else "")
