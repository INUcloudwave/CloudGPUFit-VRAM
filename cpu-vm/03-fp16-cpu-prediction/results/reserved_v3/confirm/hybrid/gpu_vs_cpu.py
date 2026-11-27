"""For fetched INUGPU traces: (1) replay the GPU's own request stream (descending, effective capacity) -> does it
reproduce the GPU measure_0 reserved? (2) replay the CPU stream (+ cuBLAS ws at probed position) -> v3 value.
(3) warmup_0 request-size multiset differences GPU vs CPU (top by bytes)."""
import sys, json
from collections import Counter
OUT = sys.argv[1]; CIDS = sys.argv[2:]
sys.argv = [sys.argv[0], "/dev/null"]
exec(open("/out/verify_guide.py").read().split("\ndef main():")[0])
pos = {}
for l in open("/results_v3/gemm_exact_positions.jsonl"):
    r = json.loads(l); pos[r["case_id"]] = r.get("ws_insert_rel")
EFF = 24979374080
for cid in CIDS:
    gs, gm, gpu = gpu_stream(Path("/f/rtx3090__trace_models/" + cid))
    root = "main395_l256"
    cs, cm, _ = cpu_stream(Path(f"/runs/{root}/cases/{cid}"))
    rg = replay(gs, gm, EFF, "descending")
    inj = [(cm["warmup_0_begin"] + pos[cid], 8519680)] if pos.get(cid) is not None else []
    rc = replay(cs, cm, EFF, "descending", inject=inj)
    g = Counter(s for a, s, k in window(gs, gm, "warmup_0") if a); c = Counter(s for a, s, k in window(cs, cm, "warmup_0") if a)
    diff = sorted(((s, g[s], c[s]) for s in set(g) | set(c) if g[s] != c[s]), key=lambda x: -abs(x[1] - x[2]) * x[0])[:10]
    row = {"cid": cid, "gpu_m0_start_reserved_MiB": gpu["m0_start_reserved"] / MiB, "gpu_peak_alloc_MiB": gpu["m0_peak_allocated"] / MiB,
           "replay_gpu_stream_MiB": None if rg["oom"] else rg["m0_start_reserved"] / MiB,
           "replay_cpu_stream_MiB": None if rc["oom"] else rc["m0_start_reserved"] / MiB,
           "gpu_max_alloc_MiB": rg.get("max_allocated", 0) / MiB, "cpu_max_alloc_MiB": rc.get("max_allocated", 0) / MiB,
           "warmup0_size_diff_gpu_vs_cpu": [[round(s / MiB, 3), a, b] for s, a, b in diff]}
    print(json.dumps(row), flush=True)
    open(OUT, "a").write(json.dumps(row) + "\n")
