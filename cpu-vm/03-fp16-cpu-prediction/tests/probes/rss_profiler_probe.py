import torch, sys, time, resource, os
import torch._C._profiler as PP
from torch.autograd import ProfilerActivity
sys.path.insert(0, "/exp/src")
from cgf680 import protocol as P
import json
def rss(): 
    for l in open("/proc/self/status"):
        if l.startswith("VmRSS"): return int(l.split()[1])/2**20
mode = sys.argv[1]; th=int(sys.argv[2]); torch.set_num_threads(th)
case={"case_id":"x","model":"facebook/opt-125m","kind":"decoder_llm","dtype":"fp16","batch_size":8,"input_len":128,"new_tokens":128,"resolution":None}
pin=P.load_pins("/exp/manifest/model_pins.json")[case["model"]]
m,_=P.load_model(case,pin,"cpu"); x=P.make_host_inputs(case,m.config); kw=P.generate_kwargs(case,m.config)
print("loaded", round(rss(),2))
if mode!="none":
    cfg = PP.ProfilerConfig(PP.ProfilerState.KINETO, False, mode=="mem", False, False, False, PP._ExperimentalConfig())
    scopes = {PP.RecordScope.USER_SCOPE} if len(sys.argv) > 3 and sys.argv[3] == "user" else set()
    torch.autograd._prepare_profiler(cfg, {ProfilerActivity.CPU}); torch.autograd._enable_profiler(cfg, {ProfilerActivity.CPU}, scopes)
for i in range(int(os.environ.get("NRUNS", 2))):
    t=time.time(); o,_=P.invoke(m,case,x,kw); del o; print(mode, th, "run",i, round(time.time()-t,1),"s rss GiB", round(rss(),2), flush=True)
if mode!="none":
    t=time.time(); r=torch.autograd._disable_profiler(); print("disable", round(time.time()-t,1), round(rss(),2))
    t=time.time(); r.save("/tmp/t.json"); print("save", round(time.time()-t,1), round(rss(),2), os.path.getsize("/tmp/t.json")/2**30)
    import collections; d=json.load(open("/tmp/t.json")); ev=d["traceEvents"]
    print(collections.Counter((e.get("cat"), e.get("name") if e.get("cat")!="cpu_op" else "op") for e in ev).most_common(6))
