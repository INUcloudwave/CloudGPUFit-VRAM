"""GPU vs CPU request-size multiset in warmup_0 (t5-base b128/128/512, gpt2 b128/128/512, tinyllama): which sizes differ."""
import sys, json
from collections import Counter
sys.argv = [sys.argv[0], "/dev/null"]
exec(open("/out/verify_guide.py").read().split("\ndef main():")[0])
for cid, root in (("t5-base__b128__in128__new512", "main395_l256"), ("gpt2__b128__in128__new512", "main395_l256"), ("tinyllama-1.1b-chat__b128__in128__new512", "main395_l256")):
    gs, gm, _ = gpu_stream(Path("/h16/traces/" + cid)); cs, cm, _ = cpu_stream(Path(f"/runs/{root}/cases/{cid}"))
    g = Counter(s for a, s, k in window(gs, gm, "warmup_0") if a)
    c = Counter(s for a, s, k in window(cs, cm, "warmup_0") if a)
    diff = {s: (g[s], c[s]) for s in set(g) | set(c) if g[s] != c[s]}
    top = sorted(diff.items(), key=lambda kv: -abs(kv[1][0] - kv[1][1]) * kv[0])[:12]
    print(cid, "gpu allocs", sum(g.values()), "cpu allocs", sum(c.values()))
    for s, (a, b) in top:
        print(f"   size {s:>12d} ({s/2**20:8.3f} MiB) gpu {a:7d} cpu {b:7d}")
