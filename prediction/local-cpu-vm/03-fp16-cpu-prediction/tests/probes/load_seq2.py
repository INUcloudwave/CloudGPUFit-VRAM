"""GPU load sequence vs CPU-profiled fp16 load (device_map cpu), aligned with difflib."""
import sys, json, difflib
sys.path.insert(0, "/diag/src")
import torch, transformers
from cgf_diag.events import load_snapshot
import torch._C._profiler as PP
from torch.autograd import ProfilerActivity
cid, model_id, rev, st, loader = sys.argv[1:6]
d = f"/handoff/reference/traces_attn_default/{cid}"
t = json.load(open(f"{d}/trace.json")); end = next(x for x in t["history"]["boundaries"] if x["name"] == "model_load_end")
ev = load_snapshot(f"{d}/trace.snapshot.pickle.gz")["device_traces"][0]
g = []; a = f = 0
for e in ev:
    if a >= end["allocation.all.allocated"] and f >= end["allocation.all.freed"]: break
    if e["action"] == "alloc": a += 1; g.append(e["size"])
    elif e["action"] == "free_completed": f += 1; g.append(-e["size"])
cfg = PP.ProfilerConfig(PP.ProfilerState.KINETO, False, True, False, False, False, PP._ExperimentalConfig())
torch.autograd._prepare_profiler(cfg, {ProfilerActivity.CPU}); torch.autograd._enable_profiler(cfg, {ProfilerActivity.CPU}, {PP.RecordScope.USER_SCOPE})
m = getattr(transformers, loader).from_pretrained(model_id, revision=rev, torch_dtype=torch.float16, low_cpu_mem_usage=True,
        device_map={"": "cpu"}, use_safetensors=(st == "1"), local_files_only=True)
m = m.to(device="cpu", dtype=torch.float16).eval()
r = torch.autograd._disable_profiler(); r.save("/tmp/l.json")
c = [e["args"]["Bytes"] for e in json.load(open("/tmp/l.json"))["traceEvents"] if e.get("name") == "[memory]" and e["args"].get("Device Type") == 0]
sm = difflib.SequenceMatcher(None, g, c, autojunk=False)
ops = [(tag, g[i1:i2][:6], c[j1:j2][:6], i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]
print(json.dumps({"cid": cid, "gpu_n": len(g), "cpu_n": len(c), "matching": sum(b.size for b in sm.get_matching_blocks()), "nonequal_ops": len(ops), "ops_head": ops[:12]}))
