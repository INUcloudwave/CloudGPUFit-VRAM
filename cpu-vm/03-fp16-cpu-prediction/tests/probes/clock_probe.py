import torch, time, json
import torch._C._profiler as PP
from torch.autograd import ProfilerActivity
cfg = PP.ProfilerConfig(PP.ProfilerState.KINETO, False, True, False, False, False, PP._ExperimentalConfig())
torch.autograd._prepare_profiler(cfg, {ProfilerActivity.CPU}); torch.autograd._enable_profiler(cfg, {ProfilerActivity.CPU}, {PP.RecordScope.USER_SCOPE})
marks=[]
for i in range(3):
    t0=(time.monotonic_ns(), time.time_ns(), time.perf_counter_ns()); x=torch.empty(777777+i, dtype=torch.uint8); t1=(time.monotonic_ns(), time.time_ns(), time.perf_counter_ns()); marks.append((t0,t1)); del x
r=torch.autograd._disable_profiler(); r.save("/tmp/c.json")
ev=[e for e in json.load(open("/tmp/c.json"))["traceEvents"] if e.get("name")=="[memory]" and e["args"]["Bytes"]>0]
ev=[e for e in ev if 777777<=e["args"]["Bytes"]<777780]
for (t0,t1),e in zip(marks,ev):
    ts=e["ts"]*1000
    print("mono", t0[0]<=ts<=t1[0], "real", t0[1]<=ts<=t1[1], "perf", t0[2]<=ts<=t1[2], ts, t0)
