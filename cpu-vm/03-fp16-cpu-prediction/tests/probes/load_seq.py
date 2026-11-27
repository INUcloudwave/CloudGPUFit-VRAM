"""Compare GPU load-phase alloc/free sequence with (a) CPU-profiled from_pretrained(device_map=cpu)."""
import sys, json, gzip, pickle, io
sys.path.insert(0, "/diag/src"); sys.path.insert(0, "/exp/src")
import torch, transformers
from cgf_diag.events import load_snapshot
import torch._C._profiler as PP
from torch.autograd import ProfilerActivity
cid, model_id, rev, st, loader = sys.argv[1:6]
d = f"/handoff/reference/traces_attn_default/{cid}"
t = json.load(open(f"{d}/trace.json")); end = next(x for x in t["history"]["boundaries"] if x["name"] == "model_load_end")
ev = load_snapshot(f"{d}/trace.snapshot.pickle.gz")["device_traces"][0]
gseq = []; a = f = 0
for e in ev:
    if a >= end["allocation.all.allocated"] and f >= end["allocation.all.freed"]: break
    if e["action"] == "alloc": a += 1; gseq.append(e["size"])
    elif e["action"] == "free_completed": f += 1; gseq.append(-e["size"])
cfg = PP.ProfilerConfig(PP.ProfilerState.KINETO, False, True, False, False, False, PP._ExperimentalConfig())
torch.autograd._prepare_profiler(cfg, {ProfilerActivity.CPU}); torch.autograd._enable_profiler(cfg, {ProfilerActivity.CPU}, {PP.RecordScope.USER_SCOPE})
m = getattr(transformers, loader).from_pretrained(model_id, revision=rev, torch_dtype=torch.float16, low_cpu_mem_usage=True,
        device_map={"": "cpu"}, use_safetensors=(st == "1"), local_files_only=True)
m = m.to(device="cpu", dtype=torch.float16).eval()
r = torch.autograd._disable_profiler(); r.save("/tmp/l.json")
cseq = [e["args"]["Bytes"] for e in json.load(open("/tmp/l.json"))["traceEvents"] if e.get("name") == "[memory]" and e["args"].get("Device Type") == 0]
ga = [x for x in gseq if x > 0]; ca = [x for x in cseq if x > 0]
print(json.dumps({"cid": cid, "gpu_events": len(gseq), "gpu_allocs": len(ga), "gpu_alloc_bytes": sum(ga),
                  "cpu_events": len(cseq), "cpu_allocs": len(ca), "cpu_alloc_bytes": sum(ca),
                  "gpu_first": gseq[:12], "cpu_first": cseq[:12],
                  "gpu_allocs_wo_warmup_equal_cpu_allocs": ga[1:] == ca}))
