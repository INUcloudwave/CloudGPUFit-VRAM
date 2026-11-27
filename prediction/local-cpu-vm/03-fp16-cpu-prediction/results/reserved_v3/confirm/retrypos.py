"""t5-base b128/128/512 (retry): where does the GPU release its cache vs where xMem (GPU's own request stream,
descending, effective capacity) releases? Positions are indices in the GPU alloc/free request stream."""
import sys, json, pickle, gzip
sys.argv = [sys.argv[0], "/dev/null"]
exec(open("/out/verify_guide.py").read().split("\ndef main():")[0])
d = Path("/h16/traces/t5-base__b128__in128__new512")
T = json.loads((d / "trace.json").read_text())
ev = pickle.load(gzip.open(d / T["history"]["snapshot_file"], "rb"))["device_traces"][0]
# GPU: release bursts, indexed by number of alloc/free_completed events seen so far
n = 0; bursts = []; last = None
for e in ev:
    a = e["action"]
    if a in ("alloc", "free_completed"):
        n += 1
    elif a == "segment_free":
        if last != n:
            bursts.append([n, 0, 0]); last = n
        bursts[-1][1] += 1; bursts[-1][2] += e["size"]
gs, gm, gpu = gpu_stream(d)
sim = make_sim(24979374080, "descending")
rel = []
orig = sim.release_cached_blocks
pos = [0]
def counted():
    before = sim.total_allocated_memory
    orig()
    rel.append([pos[0], round((before - sim.total_allocated_memory) / MiB)])
sim.release_cached_blocks = counted
dev, st, live = Device(index=0), Stream(index=0), {}
marks = {v: k for k, v in gm.items()}
at = {}
for i, (a, s, k) in enumerate(gs):
    pos[0] = i
    if i in marks: at[marks[i]] = sim.total_allocated_memory // MiB
    if a: live[k] = sim.malloc(dev, st, s)
    else:
        b = live.pop(k, None)
        if b is not None: sim.free(b)
out = {"gpu_release_bursts": [[p, c, round(z / MiB)] for p, c, z in bursts], "xmem_releases": rel, "marks": gm, "xmem_reserved_at_marks_MiB": at}
json.dump(out, open("/out/retrypos.json", "w"), indent=0)
print(json.dumps({"gpu_bursts": out["gpu_release_bursts"], "xmem": rel, "marks": gm}))
